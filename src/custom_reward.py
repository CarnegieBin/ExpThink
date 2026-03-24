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
from copy import deepcopy

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict
from typing import TypedDict


class _ExperienceEntry(TypedDict):
    length: int
    response: str


class CustomRewardManager:
    """The reward manager.
    """

    # 跨调用持久化：key=prompt_str，value={"length": int, "response": str}
    experience_box: dict[str, _ExperienceEntry] = {}

    def __init__(self,
                 tokenizer,
                 num_examine,
                 compute_score=None,
                 reward_fn_key='data_source',
                 is_train: bool = True,
                 q_ratio: float = 0.1) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.is_train = is_train
        self.q_ratio = q_ratio

    def __call__(self, data: DataProto, return_dict: bool = False, is_verify=False, reward_tensor=None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']
        if "uid" not in data.non_tensor_batch.keys():
            reward_tensor = torch.zeros_like(
                data.batch['responses'], dtype=torch.float32
            )
            reward_extra_info = defaultdict(list)

            already_print_data_sources = {}

            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                eos_token = self.tokenizer.eos_token
                if response_str.endswith(eos_token):
                    response_str = response_str[:-len(eos_token)]

                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get('extra_info', None)

                result = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

                result.update({
                    "response_length": valid_response_length,
                })

                if isinstance(result, dict):
                    score = result["score"]
                    for key, value in result.items():
                        reward_extra_info[key].append(value)
                else:
                    score = result

                reward_tensor[i, valid_response_length - 1] = score

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    if isinstance(result, dict):
                        for key, value in result.items():
                            print(f"[{key}]", value)
                    else:
                        print(f"[score]", score)

            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return reward_tensor
        else:

            reward_tensor = torch.zeros_like(
                data.batch['responses'], dtype=torch.float32
            )
            reward_extra_info = defaultdict(list)

            already_print_data_sources = {}

            # ===== 新增：用于 uid 分组的缓存 =====
            uid_to_indices: dict[str, list[int]] = defaultdict(list)
            sample_correct: dict[int, bool] = {}
            sample_resp_len: dict[int, int] = {}
            sample_resp_str: dict[int, str] = {}
            sample_prompt_str: dict[int, str] = {}

            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                # ===== 取 uid（假设在 non_tensor_batch 中）=====
                uid = data_item.non_tensor_batch["uid"]
                uid_to_indices[uid].append(i)

                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                eos_token = self.tokenizer.eos_token
                if response_str.endswith(eos_token):
                    response_str = response_str[:-len(eos_token)]

                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get('extra_info', None)

                result = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

                result.update({
                    "response_length": valid_response_length,
                })

                # ===== 统一 score / correct 判定 =====
                if isinstance(result, dict):
                    score = result["score"]
                    correct = score > 0
                    for key, value in result.items():
                        reward_extra_info[key].append(value)
                else:
                    score = result
                    correct = score > 0

                # ===== 缓存样本级信息（用于第二阶段）=====
                sample_correct[i] = correct
                sample_resp_len[i] = valid_response_length
                sample_resp_str[i] = response_str
                sample_prompt_str[i] = prompt_str

                # ===== 先正常写 reward（后面可能被覆盖为 0）=====
                reward_tensor[i, valid_response_length - 1] = score

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    if isinstance(result, dict):
                        for key, value in result.items():
                            print(f"[{key}]", value)
                    else:
                        print(f"[score]", score)

            # =====================================================================
            # 第二阶段：按 uid 做组内规则修正（experience_box 动态截断）
            # =====================================================================
            for uid, indices in uid_to_indices.items():
                # 1. 找出本组内最短的正确轨迹
                correct_indices = [i for i in indices if sample_correct[i]]
                if correct_indices:
                    best_i = min(correct_indices, key=lambda i: sample_resp_len[i])
                    best_len = sample_resp_len[best_i]
                    best_resp = sample_resp_str[best_i]
                    prompt_key = sample_prompt_str[best_i]

                    # 2. 更新 experience_box
                    if prompt_key not in CustomRewardManager.experience_box:
                        CustomRewardManager.experience_box[prompt_key] = {
                            "length": best_len,
                            "response": best_resp,
                        }
                    elif best_len < CustomRewardManager.experience_box[prompt_key]["length"]:
                        CustomRewardManager.experience_box[prompt_key] = {
                            "length": best_len,
                            "response": best_resp,
                        }

                # 3. 训练模式下，用 experience_box + q 做动态截断
                if self.is_train:
                    prompt_key = sample_prompt_str[indices[0]]
                    if prompt_key in CustomRewardManager.experience_box:
                        history_best_len: int = CustomRewardManager.experience_box[prompt_key]["length"]
                        cutoff = int(history_best_len * (1 + self.q_ratio))
                        for i in indices:
                            if sample_correct[i] and sample_resp_len[i] > cutoff:
                                _ = reward_tensor[i].zero_()

            # =====================================================================

            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return reward_tensor
