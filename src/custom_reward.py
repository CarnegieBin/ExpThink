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
import numpy as np
from collections import defaultdict
from typing import TypedDict, cast


class _ExperienceEntry(TypedDict):
    cutoff_len: int     # 组内正确轨迹长度的 P(q_ratio*100) 分位数，用于 cutoff 计算
    response: str
    response_ids: list[int]  # 长度最接近 P(q_ratio*100) 的轨迹 token ids，用于前缀匹配
    pass_rate: float    # 该题通过率的 EMA，离线 64 次采样作为 step0 初始化


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
                 q_ratio: float = 0.3,
                 threshold: float = 0.4) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.is_train = is_train
        self.q_ratio = q_ratio  # cutoff 分位数（0~1），P(q_ratio*100) 直接作为截断阈值
        self.threshold: float = threshold  # 触发压缩截断所需的通过率阈值
        self.pass_rate_ema: float = 0.1  # pass_rate EMA 平滑系数；离线 64 次采样作为 step0

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
            sample_resp_ids: dict[int, torch.Tensor] = {}  # 用于前缀 mask 的 token id
            sample_reward_model: dict[int, dict[str, object]] = {}  # 用于第一 epoch 预热 experience_box

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
                sample_resp_ids[i] = valid_response_ids  # 缓存 token ids，用于前缀比较
                sample_reward_model[i] = cast(dict[str, object], data_item.non_tensor_batch.get('reward_model', {}))

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
            prefix_mask_len_list = [0] * len(data)  # 每条轨迹的前缀 mask 长度，默认 0

            for uid, indices in uid_to_indices.items():
                # 1. 找出本组内所有正确轨迹，计算组内通过率
                correct_indices = [i for i in indices if sample_correct[i]]
                prompt_key = sample_prompt_str[indices[0]]
                group_pass_rate = len(correct_indices) / len(indices)

                # 2. EMA 更新 experience_box 中的 pass_rate（离线数据作为 step0 初始值）
                if prompt_key in CustomRewardManager.experience_box:
                    old_rate = CustomRewardManager.experience_box[prompt_key]["pass_rate"]
                    CustomRewardManager.experience_box[prompt_key]["pass_rate"] = (
                        self.pass_rate_ema * group_pass_rate
                        + (1 - self.pass_rate_ema) * old_rate
                    )

                # 3a. 离线预热：prompt 首次出现且 reward_model 中携带离线统计值时初始化 experience_box
                #     仅在 experience_box 中尚无该 prompt 时执行，避免覆盖在线更新的结果
                if prompt_key not in CustomRewardManager.experience_box:
                    offline_rm = sample_reward_model.get(indices[0], {})
                    offline_p20 = offline_rm.get("offline_p20_correct_len")
                    offline_pr  = offline_rm.get("offline_pass_rate")
                    if isinstance(offline_p20, int) and isinstance(offline_pr, float):
                        CustomRewardManager.experience_box[prompt_key] = _ExperienceEntry(
                            cutoff_len=offline_p20,
                            response="",
                            response_ids=[],
                            pass_rate=offline_pr,
                        )

                # 3b. 有正确轨迹时更新 cutoff_len 和前缀参考轨迹
                if correct_indices:
                    correct_lens = [sample_resp_len[i] for i in correct_indices]

                    # P(q_ratio*100) 分位数直接作为 cutoff（鲁棒，单次侥幸短答案不会主导）
                    p_len = int(np.percentile(correct_lens, self.q_ratio * 100))

                    # 取长度最接近 P(q_ratio*100) 的轨迹作为前缀参考（有代表性，非极端短）
                    ref_i = min(correct_indices, key=lambda i: abs(sample_resp_len[i] - p_len))

                    if prompt_key not in CustomRewardManager.experience_box:
                        # 首次出现（无离线预热）：创建完整条目，pass_rate 用当前批次初始化
                        CustomRewardManager.experience_box[prompt_key] = _ExperienceEntry(
                            cutoff_len=p_len,
                            response=sample_resp_str[ref_i],
                            response_ids=[int(t.item()) for t in sample_resp_ids[ref_i]],
                            pass_rate=group_pass_rate,
                        )
                    elif p_len < CustomRewardManager.experience_box[prompt_key]["cutoff_len"]:
                        # 只更新 cutoff 相关字段，保留已 EMA 更新的 pass_rate
                        entry = CustomRewardManager.experience_box[prompt_key]
                        entry["cutoff_len"] = p_len
                        entry["response"] = sample_resp_str[ref_i]
                        entry["response_ids"] = [int(t.item()) for t in sample_resp_ids[ref_i]]

                # 4. 训练模式：以 experience_box 中的 pass_rate 判断是否触发压缩截断
                if self.is_train:
                    entry: _ExperienceEntry | None = CustomRewardManager.experience_box.get(prompt_key)
                    # 无离线预热且首批全错时 entry 为 None，fallback 到当前批次通过率
                    ref_pass_rate: float = entry["pass_rate"] if entry else group_pass_rate

                    if entry and ref_pass_rate >= self.threshold:
                        cutoff = entry["cutoff_len"]
                        for i in indices:
                            if sample_correct[i] and sample_resp_len[i] > cutoff:
                                _ = reward_tensor[i].zero_()

                    # 5. 计算 reward=0 轨迹与参考轨迹的公共前缀长度（用于梯度 mask）
                    #    参考轨迹为长度最接近 P(q_ratio*100) 的历史正确轨迹（有代表性）
                    ref_ids = None
                    if entry and "response_ids" in entry:
                        ref_ids = torch.tensor(entry["response_ids"], dtype=torch.long)

                    if ref_ids is not None:
                        for i in indices:
                            if reward_tensor[i].sum().item() == 0:
                                traj_ids = sample_resp_ids[i]
                                min_len = min(len(ref_ids), len(traj_ids))
                                common = 0
                                for j in range(min_len):
                                    if traj_ids[j].item() == ref_ids[j].item():
                                        common += 1
                                    else:
                                        break
                                prefix_mask_len_list[i] = common

            # 将前缀 mask 长度加入 reward_extra_info，供 compute_advantage 使用
            reward_extra_info["prefix_mask_len"] = prefix_mask_len_list

            # =====================================================================

            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return reward_tensor
