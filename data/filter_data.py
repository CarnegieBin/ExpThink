"""
过滤 sample_data.jsonl，生成 filter_data.jsonl

过滤逻辑（基于讨论结论）：
- 移除 pass_rate < 0.05：模型完全不会，无正确轨迹可学
- 保留 pass_rate >= 0.05（移除 data.md 中的上界 0.95）：
    高 pass_rate + 宽长度分布的题，截断机制会制造有效压缩信号
- compression_potential = p80 / p20 >= 1.5（比 data.md 的 1.3 更严，信号更纯）
    阈值推导：cutoff ≈ p20 * (1 + q_ratio)，q_ratio=0.3
    要让 p80 明显超出 cutoff，需 p80/p20 > 1 + q_ratio = 1.3，取 1.5 留余量

每条输出记录额外附加统计字段，便于后续课程学习分级或调试。
"""

import json
import numpy as np
from pathlib import Path

# ── 超参 ──────────────────────────────────────────────────────────────────────
Q_RATIO = 0.3                  # 与 custom_reward.py 保持一致
PASS_RATE_MIN = 0.05           # 低于此值：模型完全不会，移除
COMPRESSION_THRESHOLD = 1.5    # p80/p20 阈值（推导自 q_ratio，见文件头注释）
MIN_CORRECT_SAMPLES = 4        # 至少需要这么多正确样本才能稳定计算分位数
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE  = Path(__file__).parent / "sample_data.jsonl"
OUTPUT_FILE = Path(__file__).parent / "filter_data.jsonl"


def compute_stats(lengths: list[int], correct: list[bool]) -> dict:
    """计算每道题的核心统计量"""
    correct_lens = [l for l, c in zip(lengths, correct) if c]
    pass_rate = len(correct_lens) / len(lengths) if lengths else 0.0

    if len(correct_lens) < 2:
        return {
            "pass_rate": pass_rate,
            "n_correct": len(correct_lens),
            "p20_correct_len": None,
            "p80_correct_len": None,
            "mean_correct_len": None,
            "compression_potential": 0.0,
        }

    p20 = float(np.percentile(correct_lens, 20))
    p80 = float(np.percentile(correct_lens, 80))
    compression_potential = round(p80 / p20, 3) if p20 > 0 else 0.0

    return {
        "pass_rate": round(pass_rate, 4),
        "n_correct": len(correct_lens),
        "p20_correct_len": int(p20),
        "p80_correct_len": int(p80),
        "mean_correct_len": round(float(np.mean(correct_lens)), 1),
        "compression_potential": compression_potential,
    }


def assign_tier(stats: dict) -> str:
    """课程学习分级（可选，用于后续加权采样）"""
    pr = stats["pass_rate"]
    cp = stats["compression_potential"]
    if 0.1 <= pr <= 0.8 and cp >= 2.0:
        return "tier1"
    if PASS_RATE_MIN <= pr and cp >= COMPRESSION_THRESHOLD:
        return "tier2"
    return "filtered"


def should_keep(stats: dict) -> bool:
    if stats["pass_rate"] < PASS_RATE_MIN:
        return False
    if stats["n_correct"] < MIN_CORRECT_SAMPLES:
        return False
    if stats["compression_potential"] < COMPRESSION_THRESHOLD:
        return False
    return True


def main():
    total = kept = 0
    tier_counts = {"tier1": 0, "tier2": 0}

    with INPUT_FILE.open() as fin, OUTPUT_FILE.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)

            stats = compute_stats(record["length"], record["correct"])

            if not should_keep(stats):
                continue

            tier = assign_tier(stats)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            kept += 1

            # 输出：原始字段 + 统计字段
            out = {
                "question": record["question"],
                "length":   record["length"],
                "correct":  record["correct"],
                **stats,
                "tier": tier,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"总计: {total} 条")
    print(f"保留: {kept} 条  ({kept/total*100:.1f}%)")
    print(f"过滤: {total - kept} 条  ({(total-kept)/total*100:.1f}%)")
    print(f"Tier1: {tier_counts.get('tier1', 0)}, Tier2: {tier_counts.get('tier2', 0)}")


if __name__ == "__main__":
    main()
