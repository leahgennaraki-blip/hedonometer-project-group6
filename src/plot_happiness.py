# src/plot_happiness.py
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    data_path = Path("data/processed/guardian_articles_with_scores.csv")
    df = pd.read_csv(data_path)

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 只保留有 happiness 值的文章
    df = df.dropna(subset=["happiness"])

    # === 图 1：两个时期的 happiness 分布 KDE ===
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=df,
        x="happiness",
        hue="period",         # 2010-2013 vs 2020-2023
        common_norm=False,
        fill=True,
        alpha=0.4,
    )
    plt.title("Happiness score distributions by period")
    plt.xlabel("Happiness score")
    plt.ylabel("Density")
    out1 = figures_dir / "happiness_distribution_by_period.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"Saved {out1}")

    # === 图 2：按 section + period 的箱线图 ===
    # 只保留文章数量 >= 10 的 section，避免图里很多很小的类
    counts = df["section_name"].value_counts()
    valid_sections = counts[counts >= 10].index
    df_filtered = df[df["section_name"].isin(valid_sections)]

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_filtered,
        x="section_name",
        y="happiness",
        hue="period",
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Happiness by section and period")
    plt.xlabel("Section")
    plt.ylabel("Happiness score")
    plt.legend(title="Period")
    out2 = figures_dir / "happiness_by_section_and_period_boxplot.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"Saved {out2}")

    # 你也可以继续扩展：
    # - 不同年份的平均 happiness 折线图
    # - 不同 section 在两个时期的均值 + 置信区间条形图
    # 这些都会帮助你在 README 里做比较和不确定性分析[4]。


if __name__ == "__main__":
    main()