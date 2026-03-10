# src/compute_labmt_scores.py
from pathlib import Path
import re

import numpy as np
import pandas as pd

from load_labmt import load_labmt  # 按你自己的实现保持不变


TOKEN_RE = re.compile(r"[A-Za-z]+")


def tokenize(text: str):
    """把文章文本转成小写单词列表。"""
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


def main():
    # 1. 读入 process 后的文章数据
    processed_path = Path("data/processed/guardian_articles_processed.csv")
    df = pd.read_csv(processed_path)
    print("Loaded processed articles:", df.shape)

    # 2. 加载 labMT 词典
    labmt_path = Path("data/raw/Data_Set_S1.txt")
    labmt_df = load_labmt(labmt_path)

    # === 根据你实际的列名修改下面两个变量 ===
    word_col = "word"               # 单词列
    score_col = "happiness_average" # happiness 列
    # =======================================

    word_to_score = dict(zip(labmt_df[word_col], labmt_df[score_col]))
    print(f"Loaded labMT lexicon with {len(word_to_score)} words")

    # 3. 对每篇文章计算 happiness + 命中词数
    def compute_happiness(text: str):
        tokens = tokenize(text)
        scores = [word_to_score[w] for w in tokens if w in word_to_score]
        if not scores:
            return np.nan
        return float(np.mean(scores))

    def count_labmt_words(text: str):
        tokens = tokenize(text)
        return sum(1 for w in tokens if w in word_to_score)

    df["happiness"] = df["body_text"].apply(compute_happiness)
    df["labmt_token_count"] = df["body_text"].apply(count_labmt_words)

    # 4. 保存「逐文章」级别的数据（已经有）
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    scores_path = processed_dir / "guardian_articles_with_scores.csv"
    df.to_csv(scores_path, index=False, encoding="utf-8")
    print(f"Saved scores to {scores_path}")
    print(df[["section_name", "period", "happiness", "labmt_token_count"]].head())

    # 5. 生成新的 summary table（按 section + period 聚合）
    #    这一步就是你想要新增的 table。
    summary = (
        df.dropna(subset=["happiness"])
        .groupby(["section_name", "period"])
        .agg(
            n_articles=("id", "nunique")    # 或者用 ("happiness", "size") 也可以
            if "id" in df.columns
            else ("happiness", "size"),
            mean_happiness=("happiness", "mean"),
            median_happiness=("happiness", "median"),
            sd_happiness=("happiness", "std"),
            mean_labmt_tokens=("labmt_token_count", "mean"),
        )
        .reset_index()
    )

    # 6. 把 summary table 保存到 tables/ 目录
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / "guardian_section_happiness_summary.csv"
    summary.to_csv(table_path, index=False, encoding="utf-8")

    print(f"Saved summary table to {table_path}")
    print(summary.head(8))


if __name__ == "__main__":
    main()