# src/plot_happiness.py
from pathlib import Path
# from analysis import plot_ridgeline_with_stats

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def add_mean_sd(ax, data, color, label, alpha=0.4, linewidth=1.5):
    """Add vertical lines for mean and mean ± SD."""
    mean = np.mean(data)
    sd = np.std(data, ddof=1)
    ax.axvline(mean, color=color, linestyle='-', linewidth=linewidth, alpha=alpha,
               label=f'Mean ({label})')
    ax.axvline(mean - sd, color=color, linestyle=':', linewidth=linewidth, alpha=alpha,
               label=f'Mean - SD ({label})')
    ax.axvline(mean + sd, color=color, linestyle=':', linewidth=linewidth, alpha=alpha,
               label=f'Mean + SD ({label})')

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

    # === 图 3：Intersection plot ===
    # Replace boxplot\

    # =========================================
    # Test: Violin plot for comparison 3 (one per section, both periods)
    # =========================================

    # fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    # for ax, section in zip(axes, target_sections):
    #     df_section = df[df["section_name"] == section]
    #     if df_section.empty:
    #         continue
    #     sns.violinplot(data=df_section, x="happiness", y="period", hue="period",
    #                 split=True, inner="quartile", palette="Set2", ax=ax,
    #                 legend=False, orient='h', width=10, cut=0)
    #     ax.set_title(section)
    #     ax.set_ylabel("Period")
    #     ax.set_xlabel("Happiness score")
    #     ax.set_yticks([])  # remove y-ticks because periods are shown in legend
    # # Add a custom legend for the periods
    # from matplotlib.lines import Line2D
    # legend_elements = [
    #     Line2D([0], [0], color='#66c2a5', lw=4, label='2010-13'),
    #     Line2D([0], [0], color='#fc8d62', lw=4, label='2020-23')
    # ]
    # axes[0].legend(handles=legend_elements, title='Period', loc='upper right')
    # plt.tight_layout()
    # out_violin = figures_dir / "happiness_violin_by_section_horizontal_split.png"
    # plt.savefig(out_violin, dpi=300)
    # plt.close()
    # print(f"Saved split violin plot to {out_violin}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    period_colors = {'2010-2013': 'blue', '2020-2023': 'orange'}

    for ax, section in zip(axes, target_sections):
        df_section = df[df["section_name"] == section]
        if df_section.empty:
            continue
        # Plot densities for each period
        for period in df["period"].unique():
            data_period = df_section[df_section["period"] == period]["happiness"].values
            if len(data_period) >= 5:
                sns.kdeplot(x=data_period, label=period,
                            common_norm=False, fill=True, alpha=0.5, ax=ax,
                            color=period_colors[period])
        # Add mean ± SD lines (no legend labels)
        for period in df["period"].unique():
            data_period = df_section[df_section["period"] == period]["happiness"].values
            if len(data_period) >= 5:
                mean = np.mean(data_period)
                sd = np.std(data_period, ddof=1)
                ax.axvline(mean, color=period_colors[period], linestyle='-', linewidth=1.5, alpha=1)
                ax.axvline(mean - sd, color=period_colors[period], linestyle=':', linewidth=1.6, alpha=1)
                ax.axvline(mean + sd, color=period_colors[period], linestyle=':', linewidth=1.6, alpha=1)
        # Set labels with larger font
        ax.set_title(section, fontsize=14)
        ax.set_xlabel("Happiness score", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.tick_params(axis='both', labelsize=12)

    # Create a combined legend for the top subplot
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_handles = [
        Patch(facecolor=period_colors['2010-2013'], alpha=0.4, label='2010-13'),
        Patch(facecolor=period_colors['2020-2023'], alpha=0.4, label='2020-23'),
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.2, label='Mean'),
        Line2D([0], [0], color='black', linestyle=':', linewidth=1.2, label='Mean ± SD')
    ]
    axes[0].legend(handles=legend_handles, title='Legend', loc='upper left', fontsize=11)

    plt.tight_layout()
    out3 = figures_dir / "happiness_by_section_and_period_boxplot.png.png"
    plt.savefig(out3, dpi=300)
    plt.close()
    print(f"Saved {out3}")

    # # Plot 3: Boxplot by section and period (using filtered data)
    # if not df_sections_filtered.empty:
    #     plt.figure(figsize=(10, 6))
    #     sns.boxplot(
    #         data=df_sections_filtered,
    #         x="section_name",
    #         y="happiness",
    #         hue="period",
    #     )
    #     plt.xticks(rotation=45, ha="right")
    #     plt.title("Happiness by section and period (World news, Politics, Opinion)")
    #     plt.xlabel("Section")
    #     plt.ylabel("Happiness score")
    #     plt.legend(title="Period")
    #     out3 = figures_dir / "happiness_by_section_and_period_boxplot.png"
    #     plt.tight_layout()
    #     plt.savefig(out3, dpi=300)
    #     plt.close()
    #     print(f"Saved {out3}")
    # else:
    #     print("Not enough data for boxplot.")

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