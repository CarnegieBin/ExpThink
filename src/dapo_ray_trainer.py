# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
import os
import json
import shutil
import glob as glob_module
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    _timer,
    apply_kl_penalty,
)
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

# ── Monkey-patch DataProto.concat ────────────────────────────────────────────
# When enable_chunked_prefill=True, different DP workers may pad generated
# sequences to slightly different lengths. The original concat crashes with:
#   "Sizes of tensors must match except in dimension 0"
# This patch aligns sequence-length dims (dim≥1) across all shards before cat.
_orig_dataproto_concat = DataProto.concat

@staticmethod
def _patched_dataproto_concat(data):
    if data and data[0].batch is not None:
        all_keys = list(data[0].batch.keys())
        for key in all_keys:
            tensors = [d.batch[key] for d in data]
            shapes = [t.shape for t in tensors]
            # Only act when shapes differ beyond dim 0
            if len(set(s[1:] for s in shapes)) > 1:
                ndim = tensors[0].ndim
                max_sizes = [max(s[dim] for s in shapes) for dim in range(ndim)]
                for d, t in zip(data, tensors):
                    if list(t.shape) != max_sizes:
                        # Build F.pad args: (last_dim_left, last_dim_right, ..., dim1_left, dim1_right)
                        pad_args = []
                        for dim in range(ndim - 1, 0, -1):
                            pad_args.extend([0, max_sizes[dim] - t.shape[dim]])
                        d.batch[key] = torch.nn.functional.pad(t, pad_args, value=0)
    return _orig_dataproto_concat(data)

DataProto.concat = _patched_dataproto_concat
# ─────────────────────────────────────────────────────────────────────────────


def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: np.ndarray,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_response_mask(data: DataProto):
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch['response_mask'] = compute_response_mask(data)
    # prepare response group

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=data.batch['token_level_rewards'],
        response_mask=data.batch['response_mask'],
        index=data.non_tensor_batch['uid'])
    data.batch['advantages'] = advantages
    data.batch['returns'] = returns

    return data


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def _init_trajectory_log_dir(self, resumed_from_step: int = 0):
        """
        Initialize the trajectory log directory under <cwd>/log_data/<project>/<exp>/.

        - resumed_from_step == 0: fresh start, wipe the entire directory if it exists.
        - resumed_from_step > 0:  recovering from a checkpoint at that step, so keep
          files for steps <= resumed_from_step and delete everything after it.

        Sub-directories:
          train/  -> one file per training step: step_{N:06d}.jsonl
          val/    -> one file per validation call: step_{N:06d}.jsonl
        """
        project_name = self.config.trainer.project_name
        exp_name = self.config.trainer.experiment_name
        base_dir = os.path.join(os.getcwd(), "log_data", project_name, exp_name)

        if resumed_from_step == 0:
            # Fresh run: wipe everything and recreate
            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)
            os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "val"), exist_ok=True)
            print(f"[TrajectoryLog] Initialized fresh log directory: {base_dir}")
        else:
            # Resuming: delete any step files strictly greater than resumed_from_step
            os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "val"), exist_ok=True)
            for sub in ("train", "val"):
                pattern = os.path.join(base_dir, sub, "step_*.jsonl")
                for fpath in glob_module.glob(pattern):
                    fname = os.path.basename(fpath)          # e.g. step_000125.jsonl
                    try:
                        step_num = int(fname.replace("step_", "").replace(".jsonl", ""))
                    except ValueError:
                        continue
                    if step_num > resumed_from_step:
                        os.remove(fpath)
                        print(f"[TrajectoryLog] Removed stale log (step {step_num} > {resumed_from_step}): {fpath}")
            print(f"[TrajectoryLog] Resumed log directory at step {resumed_from_step}: {base_dir}")

        self._traj_log_base_dir = base_dir

    def _save_trajectories(self, split: str, step: int, inputs: list, outputs: list, scores: list,
                           extra: dict = None):
        """
        Save trajectory records to  <base_dir>/<split>/step_{step:06d}.jsonl.
        Each line is a JSON object with keys: input, output, score, [extra fields].
        """
        out_path = os.path.join(self._traj_log_base_dir, split, f"step_{step:06d}.jsonl")
        extra = extra or {}
        with open(out_path, "w", encoding="utf-8") as f:
            for i, (inp, out, sc) in enumerate(zip(inputs, outputs, scores)):
                record = {"step": step, "idx": i, "input": inp, "output": out, "score": sc}
                for k, v in extra.items():
                    vals = v if isinstance(v, (list, np.ndarray)) else []
                    record[k] = vals[i] if i < len(vals) else None
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[TrajectoryLog] Saved {len(inputs)} trajectories → {out_path}")

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')

        print(f'local_global_step_folder: {local_global_step_folder}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')

        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = self.config.trainer.get('max_actor_ckpt_to_keep',
                                                         None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get('max_critic_ckpt_to_keep',
                                                          None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

        # Save HF-format checkpoint for the current step so that old checkpoints can later
        # be stripped down to just their huggingface/ subdirectory.
        self.actor_rollout_wg.save_hf_checkpoint(actor_local_path)

        import glob, shutil

        # ---- Convert old checkpoints to HF format and remove non-model-weight files ----
        for ckpt_dir in glob.glob(os.path.join(self.config.trainer.default_local_dir, "global_step_*")):
            if os.path.basename(ckpt_dir) == f'global_step_{self.global_steps}':
                continue  # skip the checkpoint we just saved

            # Remove dataloader state (not model weights)
            data_pt = os.path.join(ckpt_dir, 'data.pt')
            if os.path.exists(data_pt):
                try:
                    os.remove(data_pt)
                    print(f'Removed old dataloader state: {data_pt}')
                except Exception as e:
                    print(f'Failed to remove {data_pt}: {e}')

            for role in ['actor', 'critic']:
                role_dir = os.path.join(ckpt_dir, role)
                if not os.path.isdir(role_dir):
                    continue

                hf_dir = os.path.join(role_dir, 'huggingface')
                if not os.path.isdir(hf_dir):
                    print(f'Warning: {role_dir} has no huggingface/ subdir, skipping cleanup')
                    continue

                # Delete everything inside role_dir except the huggingface/ subdirectory
                for item in os.listdir(role_dir):
                    if item == 'huggingface':
                        continue
                    item_path = os.path.join(role_dir, item)
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                        print(f'Removed old non-HF file: {item_path}')
                    except Exception as e:
                        print(f'Failed to remove {item_path}: {e}')


    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        # Collect actual response lengths (excluding padding)
        sample_response_lengths = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # Compute actual response lengths by counting non-padding tokens in the response
            # attention_mask covers the full sequence; responses occupy the last response_len positions
            response_len = output_ids.shape[1]
            if 'attention_mask' in test_output_gen_batch.batch:
                resp_attention_mask = test_output_gen_batch.batch['attention_mask'][:, -response_len:]
            else:
                # Fallback: treat all non-pad-token positions as valid
                pad_token_id = self.tokenizer.pad_token_id
                resp_attention_mask = (output_ids != pad_token_id).long()
            actual_lengths = resp_attention_mask.sum(dim=-1).cpu().tolist()
            sample_response_lengths.extend(actual_lengths)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # ---- Save all validation trajectories to disk ----
        if hasattr(self, '_traj_log_base_dir'):
            extra_info = {k: v for k, v in reward_extra_infos_dict.items() if k != "reward"}
            self._save_trajectories(
                split="val",
                step=self.global_steps,
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                extra=extra_info if extra_info else None,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        metric_dict = {}

        # ---------- PASS@k and length metrics per data source ----------
        # Determine which score variable to use for correctness
        score_key = "acc" if "acc" in reward_extra_infos_dict else "reward"
        score_vals = reward_extra_infos_dict[score_key]

        # Group by (data_source, prompt): collect per-question scores and lengths
        ds_prompt2scores: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        ds_prompt2lengths: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for idx, (ds, prompt) in enumerate(zip(data_sources, sample_inputs)):
            ds_prompt2scores[ds][prompt].append(score_vals[idx])
            ds_prompt2lengths[ds][prompt].append(sample_response_lengths[idx])

        def _pass_at_k(n: int, c: int, k: int) -> float:
            """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
            if n - c < k:
                return 1.0
            # Use logarithms to avoid overflow with large factorials
            from math import comb
            return 1.0 - comb(n - c, k) / comb(n, k)

        for ds in ds_prompt2scores:
            prompt2scores = ds_prompt2scores[ds]
            prompt2lengths = ds_prompt2lengths[ds]

            # Infer n from the number of repetitions per prompt (should be consistent)
            n_values = [len(v) for v in prompt2scores.values()]
            n = n_values[0] if n_values else 1

            # Collect per-question mean accuracy and pass@k accumulators
            per_question_mean_acc = []
            pass_k_accum: dict[int, list] = defaultdict(list)

            # Build list of k values: 1, 2, 4, ..., up to n (powers of 2)
            ks = []
            k = 1
            while k <= n:
                ks.append(k)
                k *= 2

            for prompt, scores in prompt2scores.items():
                n_q = len(scores)
                # Convert scores to binary correctness (score > 0 means correct)
                correct = [1 if s > 0 else 0 for s in scores]
                c = sum(correct)

                per_question_mean_acc.append(np.mean(correct))

                for ki in ks:
                    if ki <= n_q:
                        pass_k_accum[ki].append(_pass_at_k(n_q, c, ki))

            # Aggregate per-question length mean
            all_lengths = [l for lengths in prompt2lengths.values() for l in lengths]
            length_mean = float(np.mean(all_lengths)) if all_lengths else 0.0

            # Mean accuracy across all questions (mean of per-question means)
            mean_acc = float(np.mean(per_question_mean_acc)) if per_question_mean_acc else 0.0

            print(f"[val] data_source={ds}, n={n}, mean_acc={mean_acc:.4f}, length_mean={length_mean:.1f}")
            for ki in ks:
                pass_k_val = float(np.mean(pass_k_accum[ki])) if pass_k_accum[ki] else 0.0
                print(f"  pass@{ki}={pass_k_val:.4f}")

            # Write into metric_dict
            metric_dict[f"val-core/{ds}/pass_k/mean@{n}"] = mean_acc
            for ki in ks:
                pass_k_val = float(np.mean(pass_k_accum[ki])) if pass_k_accum[ki] else 0.0
                metric_dict[f"val-core/{ds}/pass_k/pass@{ki}"] = pass_k_val
            metric_dict[f"val-core/{ds}/response_length/mean"] = length_mean

        return metric_dict

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # Initialize trajectory log directory.
        # If global_steps > 0 here, we are resuming from a checkpoint at that step.
        self._init_trajectory_log_dir(resumed_from_step=self.global_steps)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        _traj_buffer: list = []  # accumulates trajectories across sub-batches for one training step
        print("Starting training loop...")
        for epoch in range(self.config.trainer.total_epochs):
            print("Starting epoch {}".format(epoch))
            for batch_dict in tqdm(self.train_dataloader):
                metrics = {}
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                # pop those keys for generation
                if 'multi_modal_inputs' in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                print("Generating new batch...")

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    # Decode training inputs/outputs now; scores will be appended after reward computation
                    try:
                        _train_input_ids = gen_batch.batch.get('input_ids')
                        _train_output_ids = gen_batch_output.batch.get('responses')
                        _train_inputs = (
                            [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in _train_input_ids]
                            if _train_input_ids is not None else []
                        )
                        _train_outputs = (
                            [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in _train_output_ids]
                            if _train_output_ids is not None else []
                        )
                    except Exception as _e:
                        print(f"[TrajectoryLog] Warning: failed to decode train trajectories at step "
                              f"{self.global_steps}: {_e}")
                        _train_inputs, _train_outputs = [], []

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    # print(f" 4 new_batch non_tensor_batch keys: {new_batch.non_tensor_batch.keys()}")
                    new_batch.non_tensor_batch['uid'] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    new_batch = new_batch.union(gen_batch_output)

                    with _timer('reward', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        # print("Calculating reward...")
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result['reward_tensor']
                            reward_extra_infos_dict = reward_result['reward_extra_info']
                        except Exception as e:
                            print(f'Error in reward_fn: {e}')
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({
                                k: np.array(v) for k, v in reward_extra_infos_dict.items()
                            })
                        new_batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch,
                                                                     kl_ctrl=self.kl_ctrl_in_reward,
                                                                     kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(
                                kl_metrics)
                        else:
                            new_batch.batch['token_level_rewards'] = new_batch.batch['token_level_scores']
                        # exit(0)

                    # ---- Accumulate training trajectories into buffer (all sub-batches) ----
                    if _train_inputs and hasattr(self, '_traj_log_base_dir'):
                        try:
                            # new_batch has repeated rows (rollout.n copies per prompt).
                            _n_orig = len(_train_inputs)
                            _n_repeat = self.config.actor_rollout_ref.rollout.n
                            _all_scores = new_batch.batch['token_level_scores'].sum(dim=-1).cpu().tolist()

                            for orig_i in range(_n_orig):
                                for rep_j in range(_n_repeat):
                                    flat_idx = orig_i * _n_repeat + rep_j
                                    inp = _train_inputs[orig_i]
                                    try:
                                        resp_ids = new_batch.batch['responses'][flat_idx]
                                        out = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                                    except Exception:
                                        out = _train_outputs[orig_i] if orig_i < len(_train_outputs) else ""
                                    sc = _all_scores[flat_idx] if flat_idx < len(_all_scores) else 0.0
                                    extra_fields = {
                                        ek: (ev[flat_idx] if flat_idx < len(ev) else None)
                                        for ek, ev in reward_extra_infos_dict.items()
                                    }
                                    _traj_buffer.append({"input": inp, "output": out, "score": sc, **extra_fields})
                        except Exception as _e:
                            print(f"[TrajectoryLog] Warning: failed to accumulate train trajectories at step "
                                  f"{self.global_steps}: {_e}")

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch['token_level_rewards'].sum(
                                dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch['token_level_scores'].sum(
                                dim=-1).numpy()
                        elif metric_name == "acc":
                            new_batch.non_tensor_batch["acc"] = new_batch.batch['token_level_scores'].sum(
                                dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch['uid'],
                                                   new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        # ---- stats: count degenerate 0/1 groups ----
                        num_all_zero_groups = 0
                        num_all_one_groups = 0
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            vals = np.array(metric_vals)
                            if np.all(vals == 0):
                                num_all_zero_groups += 1
                            if np.all(vals == 1):
                                num_all_one_groups += 1
                        total_groups = len(prompt_uid2metric_vals)
                        print(f"[no_filter] gen_batch {num_gen_batches}: "
                              f"total_groups={total_groups}, "
                              f"all_zero_groups={num_all_zero_groups}, "
                              f"all_one_groups={num_all_one_groups}")
                        metrics["train/all_zero_groups"] = num_all_zero_groups
                        metrics["train/all_one_groups"] = num_all_one_groups

                    else:  # NOTE: When prompts after filtering is less than train batch size, we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch['token_level_rewards'].sum(
                                dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch['token_level_scores'].sum(
                                dim=-1).numpy()
                        elif metric_name == "acc":
                            new_batch.non_tensor_batch["acc"] = new_batch.batch['token_level_scores'].sum(
                                dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch['uid'],
                                                   new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        # ---- stats: count degenerate 0/1 groups before filtering ----
                        num_all_zero_groups = 0
                        num_all_one_groups = 0
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            vals = np.array(metric_vals)
                            if np.all(vals == 0):
                                num_all_zero_groups += 1
                            if np.all(vals == 1):
                                num_all_one_groups += 1
                        total_groups = len(prompt_uid2metric_vals)
                        print(f"[filter_groups] gen_batch {num_gen_batches}: "
                              f"total_groups={total_groups}, "
                              f"all_zero_groups={num_all_zero_groups}, "
                              f"all_one_groups={num_all_one_groups}")
                        metrics["train/all_zero_groups"] = num_all_zero_groups
                        metrics["train/all_one_groups"] = num_all_one_groups

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch['uid']):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f'{num_gen_batches=}. Keep generating...')
                                continue
                            else:
                                raise ValueError(
                                    f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # ---- Flush all accumulated trajectories to disk for this step ----
                    if _traj_buffer and hasattr(self, '_traj_log_base_dir'):
                        try:
                            out_path = os.path.join(self._traj_log_base_dir, "train",
                                                     f"step_{self.global_steps:06d}.jsonl")
                            with open(out_path, "w", encoding="utf-8") as _f:
                                for _i, _rec in enumerate(_traj_buffer):
                                    _record = {"step": self.global_steps, "idx": _i,
                                               "input": _rec.pop("input"),
                                               "output": _rec.pop("output"),
                                               "score": _rec.pop("score"),
                                               **_rec}
                                    _f.write(json.dumps(_record, ensure_ascii=False) + "\n")
                            print(f"[TrajectoryLog] Saved {len(_traj_buffer)} trajectories "
                                  f"({num_gen_batches} sub-batches) → {out_path}")
                        except Exception as _e:
                            print(f"[TrajectoryLog] Warning: failed to flush train trajectories at step "
                                  f"{self.global_steps}: {_e}")

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    test_freq = self.config.trainer.test_freq
                    save_freq = self.config.trainer.save_freq
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                            (is_last_step or self.global_steps % test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or
                                                              self.global_steps % save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                #
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                _traj_buffer = []  # reset for next step


                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
