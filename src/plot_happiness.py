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
    # Keep only articles with a valid happiness score
    df = df.dropna(subset=["happiness"])

    # --- NEW: 只保留三个目标版块: World news, Politics, Opinion ---
    # --- Keep only the three target sections ---
    target_sections = ["World news", "Politics", "Opinion"]
    df = df[df["section_name"].isin(target_sections)].copy()
    print("Articles per section after filtering:")
    print(df["section_name"].value_counts())
    print()

    # === 图 1：两个时期的 happiness 分布 KDE（基于过滤后的数据） ===
    # Plot 1: KDE of happiness by period (using filtered data)
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=df,
        x="happiness",
        hue="period",         # 2010-2013 vs 2020-2023
        common_norm=False,
        fill=True,
        alpha=0.4,
    )
    plt.title("Happiness score distributions by period (World news, Politics, Opinion)")
    plt.xlabel("Happiness score")
    plt.ylabel("Density")
    out1 = figures_dir / "happiness_distribution_by_period.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"Saved {out1}")

    # === 图 2：三个版块的 happiness 分布 KDE（不分时期） ===
    # Plot 2: KDE of happiness by section (ignoring period)
    # 检查每个版块是否有足够数据（至少10篇文章）
    # Check that each section has at least 10 articles
    section_counts = df["section_name"].value_counts()
    valid_sections = section_counts[section_counts >= 10].index.tolist()
    df_sections_filtered = df[df["section_name"].isin(valid_sections)]

    if not df_sections_filtered.empty:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(
            data=df_sections_filtered,
            x="happiness",
            hue="section_name",
            common_norm=False,
            fill=True,
            alpha=0.4,
        )
        plt.title("Happiness score distributions by section (World news, Politics, Opinion)")
        plt.xlabel("Happiness score")
        plt.ylabel("Density")
        out2 = figures_dir / "happiness_distribution_by_section.png"
        plt.tight_layout()
        plt.savefig(out2, dpi=300)
        plt.close()
        print(f"Saved {out2}")
    else:
        print("Not enough data for section density plot (World news/Politics/Opinion).")

    # === 图 3：按 section + period 的箱线图（基于过滤后的数据） ===
    # Plot 3: Boxplot by section and period (using filtered data)
    if not df_sections_filtered.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df_sections_filtered,
            x="section_name",
            y="happiness",
            hue="period",
        )
        plt.xticks(rotation=45, ha="right")
        plt.title("Happiness by section and period (World news, Politics, Opinion)")
        plt.xlabel("Section")
        plt.ylabel("Happiness score")
        plt.legend(title="Period")
        out3 = figures_dir / "happiness_by_section_and_period_boxplot.png"
        plt.tight_layout()
        plt.savefig(out3, dpi=300)
        plt.close()
        print(f"Saved {out3}")
    else:
        print("Not enough data for boxplot.")

    # （可选）之前的条形图现在不再需要，已注释掉
    # (Optional) The previous bar plot is no longer needed – commented out
    """
    # === 图 3（旧）：三个版块的平均 happiness 条形图（带误差棒） ===
    # Plot 3 (old): Bar plot of mean happiness with error bars
    if not df_sections_filtered.empty:
        stats = df_sections_filtered.groupby("section_name")["happiness"].agg(["mean", "std", "count"])
        stats["ci"] = 1.96 * stats["std"] / stats["count"]**0.5  # 95% CI
        plt.figure(figsize=(8, 5))
        x_pos = range(len(stats))
        plt.bar(x_pos, stats["mean"], yerr=stats["ci"], capsize=5, alpha=0.7, color="skyblue", edgecolor="black")
        plt.xticks(x_pos, stats.index)
        plt.title("Mean happiness by section (with 95% CI)")
        plt.xlabel("Section")
        plt.ylabel("Mean happiness score")
        out3_old = figures_dir / "happiness_mean_by_section.png"
        plt.tight_layout()
        plt.savefig(out3_old, dpi=300)
        plt.close()
        print(f"Saved {out3_old}")
    """

    # 你也可以继续扩展：
    # - 不同年份的平均 happiness 折线图
    # - 不同 section 在两个时期的均值 + 置信区间条形图
    # 这些都会帮助你在 README 里做比较和不确定性分析[4]。

    # You can also continue to expand:
    # # - Line graph of average happiness in different years
    # # - Bar graph of mean + confidence interval for different sections in two periods
    # # These will help you make comparisons and uncertainty analysis in the README[4].

if __name__ == "__main__":
    main()