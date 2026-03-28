# ExpThink 训练数据处理策略

## 核心目标

在**保持准确率不变**的前提下，训练模型压缩思维链长度。

数据处理的核心逻辑：选出"有压缩空间"的题目，剔除"学不了"和"已经会了"的题目。

---

## 离线推理数据说明

每道题 64 次采样，每条记录包含：

```
{
    "prompt":   str,   # 原始问题
    "length":   int,   # 该次回答的 token 数
    "correct":  bool   # 该次回答是否正确
}
```

---

## 统计量计算

对每道题汇聚 64 条记录，计算以下统计量：

| 统计量 | 计算方式 | 含义 |
|--------|---------|------|
| `pass_rate` | `n_correct / 64` | 题目难度的代理指标 |
| `p20_correct_len` | 正确轨迹长度的 P20 | experience_box 初始化用 |
| `p80_correct_len` | 正确轨迹长度的 P80 | 模型"冗余上限" |
| `compression_potential` | `p80 / p20`（仅对正确轨迹） | 压缩空间大小 |
| `mean_correct_len` | 正确轨迹长度均值 | 当前模型效率基线 |

```python
import numpy as np
from collections import defaultdict

def compute_stats(records: list[dict]) -> dict:
    correct_lens = [r["length"] for r in records if r["correct"]]
    pass_rate = len(correct_lens) / len(records)

    if len(correct_lens) < 2:
        return {"pass_rate": pass_rate, "compression_potential": 0.0}

    p20 = np.percentile(correct_lens, 20)
    p80 = np.percentile(correct_lens, 80)

    return {
        "pass_rate": pass_rate,
        "p20_correct_len": int(p20),
        "p80_correct_len": int(p80),
        "mean_correct_len": float(np.mean(correct_lens)),
        "compression_potential": round(p80 / p20, 3) if p20 > 0 else 0.0,
    }
```

---

## 过滤策略

### 第一步：按通过率过滤（准确率保障）

```
pass_rate < 0.05  →  移除（模型完全不会，无正向信号，只会污染梯度）
pass_rate > 0.95  →  移除（模型已掌握，且大概率已有短解，无压缩训练价值）
保留区间：[0.05, 0.95]
```

**为什么移除 pass_rate > 0.95**：这类题模型已能稳定答对，P20 已经很低，
training 时 experience_box 的 cutoff 会非常紧，给不了有效压缩信号；
且这些题占训练时间却几乎不贡献 reward 变化，拖慢整体进度。

### 第二步：按压缩空间过滤（加速训练的核心）

```
compression_potential < 1.3  →  移除（正确解的长度高度集中，无从压缩）
compression_potential ≥ 1.3  →  保留
```

`compression_potential = p80 / p20` 反映了同一道题上，
模型有时用长解（p80）、有时用短解（p20）的倍数差。
差距越大，说明存在"能短但没有被激励去短"的空间，是压缩训练最有效的题目。

### 过滤后数据分级（用于课程学习，可选）

| 等级 | 条件 | 训练价值 |
|------|------|---------|
| **Tier 1** | `pass_rate ∈ [0.1, 0.8]` 且 `compression_potential ≥ 2.0` | 最高：有学习空间 + 大压缩潜力 |
| **Tier 2** | `pass_rate ∈ [0.05, 0.95]` 且 `compression_potential ∈ [1.3, 2.0)` | 中等：正常压缩训练 |
| 移除 | 其余 | - |

---

## experience_box 预热

过滤完成后，用 Tier 1 + Tier 2 数据预填充 `experience_box`，
使训练第一步就有稳定的 `cutoff_len`，避免冷启动震荡：

```python
def build_experience_box(stats_by_prompt: dict[str, dict]) -> dict:
    """
    stats_by_prompt: {prompt_str: {"p20_correct_len": int, ...}}
    注意：无 response_ids（离线未保存），前缀掩码在在线训练中自然填充
    """
    experience_box = {}
    for prompt, stats in stats_by_prompt.items():
        if stats["pass_rate"] < 0.05 or stats["pass_rate"] > 0.95:
            continue
        if stats.get("compression_potential", 0) < 1.3:
            continue
        experience_box[prompt] = {
            "cutoff_len": stats["p20_correct_len"],
            "response": "",        # 离线未保存，在线训练时覆盖
            "response_ids": [],    # 同上，前缀掩码退化为 0 直到在线填充
        }
    return experience_box
```

---

## 预期效果

| 处理步骤 | 目的 |
|---------|------|
| 移除 pass_rate < 0.05 | 消除无效梯度，保护准确率 |
| 移除 pass_rate > 0.95 | 减少对已掌握题目的无效训练，加快收敛 |
| 移除 compression_potential < 1.3 | 聚焦有压缩空间的题目，压缩信号更密集 |
| experience_box 预热 | 消除冷启动，cutoff 从第一步就稳定有效 |
