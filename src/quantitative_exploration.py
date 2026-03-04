from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 从之前写好的脚本中导入 load_labmt()
# 确保 src/load_labmt.py 里有一个名为 load_labmt 的函数
from load_labmt import load_labmt


# -----------------------------------------------------------------------------
# 路径 & 小工具函数
# -----------------------------------------------------------------------------

# 项目根目录：.../hedonometer-project-group6
ROOT = Path(__file__).resolve().parents[1]

FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


def print_section(title: str) -> None:
    """在终端里打印分隔线，让输出更清楚。"""
    bar = "=" * 80
    print("\n" + bar)
    print(title)
    print(bar)


def save_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """把 DataFrame 存到 tables/ 目录，并打印保存路径。"""
    out_path = TABLES_DIR / filename
    df.to_csv(out_path, index=index)
    print(f"Saved table: {out_path}")


def save_figure(filename: str, dpi: int = 200) -> None:
    """把当前 matplotlib 图像保存到 figures/ 目录，并打印保存路径。"""
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=dpi)
    print(f"Saved figure: {out_path}")


# -----------------------------------------------------------------------------
# 2.1 Distribution of happiness scores
# -----------------------------------------------------------------------------

def analyse_happiness_distribution(df: pd.DataFrame) -> None:
    """
    对 happiness_average 做分布分析：
    - 计算基本统计量（count, mean, median, std, p05, p95）
    - 画直方图并保存到 figures/
    - 把统计量存到 tables/
    """
    print_section("2.1 Distribution of happiness_average")

    # 取出 happiness_average 列，并去掉缺失值
    h = df["happiness_average"].dropna()

    # 1) 用 DataFrame 形式整理 summary statistics（方便保存和打印）
    summary_stats = pd.DataFrame(
        {
            "metric": [
                "count",
                "mean",
                "median",
                "std",
                "p05 (5th percentile)",
                "p95 (95th percentile)",
            ],
            "value": [
                float(h.shape[0]),
                float(h.mean()),
                float(h.median()),
                float(h.std()),
                float(h.quantile(0.05)),
                float(h.quantile(0.95)),
            ],
        }
    )

    print(summary_stats.to_string(index=False))
    save_csv(summary_stats, "happiness_average_summary_stats.csv", index=False)

    # 2) 直方图：描述 happiness_average 的分布
    plt.figure()
    plt.hist(h, bins=40, edgecolor="black")
    plt.title("Distribution of happiness_average (labMT 1.0)")
    plt.xlabel("happiness_average (1–9)")
    plt.ylabel("number of words")
    plt.tight_layout()
    save_figure("happiness_average_hist.png")
    plt.close()


# -----------------------------------------------------------------------------
# 2.2 Disagreement: contested words
# -----------------------------------------------------------------------------

def analyse_disagreement(df: pd.DataFrame) -> None:
    """
    使用 happiness_standard_deviation 分析“争议度”：
    - 画散点图：x = happiness_average, y = happiness_standard_deviation
    - 找出标准差最高的 15 个词
    """
    print_section("2.2 Disagreement: contested words")

    # 只保留这三个列，并去掉有缺失值的行
    cols = ["word", "happiness_average", "happiness_standard_deviation"]
    scatter_df = df[cols].dropna()

    # 1) 散点图：平均幸福值 vs 标准差
    plt.figure()
    plt.scatter(
        scatter_df["happiness_average"],
        scatter_df["happiness_standard_deviation"],
        s=10,
        alpha=0.35,
    )
    plt.title("Disagreement vs score: happiness_average vs standard deviation")
    plt.xlabel("happiness_average")
    plt.ylabel("happiness_standard_deviation")
    plt.tight_layout()
    save_figure("happiness_vs_std_scatter.png")
    plt.close()

    # 2) 找出标准差最大的 15 个词（争议度最高）
    most_contested_15 = (
        scatter_df.sort_values("happiness_standard_deviation", ascending=False)
        .head(15)
    )

    print("Top 15 most 'contested' words (highest std):")
    print(most_contested_15.to_string(index=False))

    save_csv(most_contested_15, "top_15_contested_words.csv", index=False)
    return most_contested_15

# -----------------------------------------------------------------------------
# 2.3 Corpus comparison: rank coverage + overlaps
# -----------------------------------------------------------------------------

def analyse_corpora(df: pd.DataFrame) -> None:
    """
    语料比较：
    - 每个语料有 rank 的词有多少？
    - 简单 overlap 表（两个语料都出现的词数）
    - 至少一个关于语料差异的图（这里用 bar chart）
    - 举一个“某语料常见但另一个语料缺失”的具体例子
    """
    print_section("2.3 Corpus comparison: corpora and ranks")

    rank_cols = ["twitter_rank", "google_rank", "nyt_rank", "lyrics_rank"]

    # 2.3 (A) 每个语料中，有 rank 的词数（即出现在该语料 top 5000 的词数）
    coverage = (
        df[rank_cols]
        .notna()           # True/False：是否有 rank
        .sum()             # 每列 True 的个数
        .reset_index()
    )
    coverage.columns = ["rank_column", "n_words_with_rank"]
    coverage["share_of_lexicon"] = coverage["n_words_with_rank"] / len(df)

    print("Words with a non-missing rank in each corpus:")
    print(coverage.to_string(index=False))
    save_csv(coverage, "corpus_rank_coverage.csv", index=False)

    # 覆盖率的柱状图
    plt.figure()
    plt.bar(coverage["rank_column"], coverage["n_words_with_rank"])
    plt.title("How many labMT words appear in each corpus top-5000?")
    plt.xlabel("corpus rank column")
    plt.ylabel("number of words with a rank")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    save_figure("corpus_rank_coverage_bar.png")
    plt.close()

    # 2.3 (B) Overlap: 语料之间的共同词
    # flags: 每一列是 True/False 表示这个词在该语料是否有 rank
    flags = pd.DataFrame(
        {
            "twitter": df["twitter_rank"].notna(),
            "google": df["google_rank"].notna(),
            "nyt": df["nyt_rank"].notna(),
            "lyrics": df["lyrics_rank"].notna(),
        }
    )

    labels = ["twitter", "google", "nyt", "lyrics"]

    # 计算每一对语料的“交集大小”：同时为 True 的行数
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            count = int((flags[a] & flags[b]).sum())
            pairs.append({"pair": f"{a}+{b}", "n_words_in_both": count})

    pairwise_overlap = pd.DataFrame(pairs).sort_values("pair")
    print("\nPairwise overlaps (words with ranks in both corpora):")
    print(pairwise_overlap.to_string(index=False))
    save_csv(pairwise_overlap, "pairwise_overlap_counts.csv", index=False)

    # 2.3 (C) 一个具体例子：
    #    “在 Twitter 中很常见，但在 NYT top-5000 里完全缺失的词”
    twitter_common_nyt_missing = (
        df[(df["twitter_rank"].notna()) & (df["nyt_rank"].isna())]
        .sort_values("twitter_rank")  # twitter_rank 越小越常见
        .head(20)[["word", "twitter_rank", "happiness_average"]]
    )

    print("\nExample: words frequent on Twitter but missing in NYT top-5000 (top 20):")
    print(twitter_common_nyt_missing.to_string(index=False))
    save_csv(twitter_common_nyt_missing, "twitter_common_nyt_missing_top20.csv", index=False)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    # 先复用之前写好的加载 + 清洗函数
    df = load_labmt()

    # 2.1 分布
    analyse_happiness_distribution(df)

    # 2.2 争议度
    analyse_disagreement(df)

    # 2.3 语料比较
    analyse_corpora(df)


if __name__ == "__main__":
    main()