"""
生成训练集 data/train.parquet

处理逻辑：
1. 读取 filter_data.jsonl，构建 question → 离线统计 的查找表
2. 读取 deepscaler.parquet
3. 按 prompt[0]['content'] 去重（保留首次出现）
4. 只保留能在 filter_data.jsonl 中匹配到的题目
5. 将离线统计值写入每行的 reward_model 字典：
       offline_pass_rate          : float  — 64次离线采样通过率（用于 experience_box 预热）
       offline_p20_correct_len    : int    — 正确轨迹长度 P20（用于 cutoff_len 初始化）
       offline_p80_correct_len    : int    — 正确轨迹长度 P80（供参考）
       offline_compression_potential: float — p80/p20（供参考）
       tier                       : str    — 课程分级 tier1/tier2
6. 保存为 data/train.parquet
"""

import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径 ──────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent
FILTER_DATA     = DATA_DIR / "filter_data.jsonl"
SOURCE_PARQUET  = DATA_DIR / "deepscaler.parquet"
OUTPUT_PARQUET  = DATA_DIR / "train.parquet"
# ─────────────────────────────────────────────────────────────────────────────


def load_filter_stats(path: Path) -> dict[str, dict]:
    """读取 filter_data.jsonl，返回 question → 离线统计 的字典"""
    stats: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec["question"]
            stats[q] = {
                "offline_pass_rate":               float(rec["pass_rate"]),
                "offline_p20_correct_len":          int(rec["p20_correct_len"]),
                "offline_p80_correct_len":          int(rec["p80_correct_len"]),
                "offline_compression_potential":    float(rec["compression_potential"]),
                "tier":                             rec["tier"],
            }
    return stats


def main():
    # 1. 读取离线统计
    print(f"读取 {FILTER_DATA.name} ...")
    filter_stats = load_filter_stats(FILTER_DATA)
    print(f"  共 {len(filter_stats)} 道题的离线统计")

    # 2. 读取 parquet
    print(f"读取 {SOURCE_PARQUET.name} ...")
    df = pd.read_parquet(SOURCE_PARQUET)
    print(f"  原始行数: {len(df)}")

    # 3. 提取 prompt content 列，用于去重和匹配
    df["_question"] = df["prompt"].apply(lambda p: p[0]["content"])

    # 4. 按 prompt content 去重（保留首次出现）
    df = df.drop_duplicates(subset="_question", keep="first")
    print(f"  去重后行数: {len(df)}")

    # 5. 只保留在 filter_data.jsonl 中有记录的题目
    mask = df["_question"].isin(filter_stats)
    df = df[mask].copy()
    print(f"  匹配 filter_data 后行数: {len(df)}")

    # 6. 将离线统计值注入 reward_model 字典
    def enrich_reward_model(row):
        rm = copy.copy(row["reward_model"])   # 浅拷贝，避免修改原始对象
        rm.update(filter_stats[row["_question"]])
        return rm

    df["reward_model"] = df.apply(enrich_reward_model, axis=1)

    # 7. 删除临时列，保存
    df = df.drop(columns=["_question"])
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\n已保存 → {OUTPUT_PARQUET}")
    print(f"最终行数: {len(df)}")

    # 8. 简单校验
    sample_rm = df.iloc[0]["reward_model"]
    print("\n[校验] 第一行 reward_model 字段:")
    for k, v in sample_rm.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
