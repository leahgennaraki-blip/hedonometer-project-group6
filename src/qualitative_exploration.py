from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 从之前写好的脚本中导入 load_labmt()
# 确保 src/load_labmt.py 里有一个名为 load_labmt 的函数
# Ensure there is function named load_labmt in src/load_labmt.py
from load_labmt import load_labmt


# -----------------------------------------------------------------------------
# Path & utility funcitons
# -----------------------------------------------------------------------------

# Project root directory：.../hedonometer-project-group6
ROOT = Path(__file__).resolve().parents[1]

FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


def print_section(title: str) -> None:
    """在终端里打印分隔线，让输出更清楚。
    Print a separator line in the terminal to make output clearer."""
    bar = "=" * 80
    print("\n" + bar)
    print(title)
    print(bar)


def save_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """把 DataFrame 存到 tables/ 目录，并打印保存路径。
    Save DataFrame to tables/ directory and print the save path."""
    out_path = TABLES_DIR / filename
    df.to_csv(out_path, index=index)
    print(f"Saved table: {out_path}")


def save_figure(filename: str, dpi: int = 200) -> None:
    """把当前 matplotlib 图像保存到 figures/ 目录，并打印保存路径。
    Save current matplotlib figure to figures/ directory and print the save path."""
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
    Analyze the distribution of happiness_average:
    - Calculate basic statistics (count, mean, median, std, p05, p95)
    - Plot histogram and save to figures/
    - Save statistics to tables/
    """
    print_section("2.1 Distribution of happiness_average")

    # 取出 happiness_average 列，并去掉缺失值
    # Extract the happiness_average column and remove missing values.
    h = df["happiness_average"].dropna()

    # 1) 用 DataFrame 形式整理 summary statistics（方便保存和打印）
    # 1) Organize summary statistics using DataFrame format (for easy saving and printing)
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
    # 2) Histogram: describe the distribution of happiness_average
    plt.figure()
    plt.hist(h, bins=40, edgecolor="black")
    plt.title("Distribution of happiness average (labMT 1.0)", fontweight='bold')
    plt.xlabel("Happiness average (1–9)")
    plt.ylabel("Number of words")
    plt.tight_layout()
    save_figure("hist_happiness_average.png")
    plt.close()


# -----------------------------------------------------------------------------
# 2.2 Disagreement: contested words
# -----------------------------------------------------------------------------

def analyse_disagreement(df: pd.DataFrame) -> None:
    """
    使用 happiness_standard_deviation 分析“争议度”：
    - 画散点图：x = happiness_average, y = happiness_standard_deviation
    - 找出标准差最高的 15 个词
    Analyze "contestedness" using happiness_standard_deviation:
    - Plot scatter plot: x = happiness_average, y = happiness_standard_deviation
    - Find the 15 words with highest standard deviation
    """
    print_section("2.2 Disagreement: contested words")

    # 只保留这三个列，并去掉有缺失值的行
    # Keep only these three columns and remove rows with missing values.
    cols = ["word", "happiness_average", "happiness_standard_deviation"]
    scatter_df = df[cols].dropna()

    # 1) 散点图：平均幸福值 vs 标准差
    # 1) Scatter plot: average happiness vs standard deviation
    plt.figure()
    plt.scatter(
        scatter_df["happiness_average"],
        scatter_df["happiness_standard_deviation"],
        s=10,
        alpha=0.35,
    )
    plt.title("Disagreement vs score: happiness average vs standard deviation", fontweight='bold')
    plt.xlabel("Happiness average")
    plt.ylabel("Happiness standard deviation")
    plt.tight_layout()
    save_figure("scatter_happiness_vs_std.png")
    plt.close()

    # 2) 找出标准差最大的 15 个词（争议度最高）
    # 2) Find the 15 words with largest standard deviation (most contested)
    most_contested_15 = (
        scatter_df.sort_values("happiness_standard_deviation", ascending=False)
        .head(15)
    )

    print("Top 15 most 'contested' words (highest std):")
    print(most_contested_15.to_string(index=False))

    save_csv(most_contested_15, "top_15_contested_words.csv", index=False)
    
    # EXTRA: Corpus unique word rankings and disagreement
    
    print_section("Extra: Disagreement and ranking plot for copus unique words")
    # Base columns we need
    base_cols = ["word", "happiness_average", "happiness_standard_deviation"]
    rank_cols = ["twitter_rank", "google_rank", "nyt_rank", "lyrics_rank"]
    
    # Corpus display names and info
    corpus_info = [
        {'name': 'Twitter', 'column': 'twitter_rank', 'color': '#1DA1F2', 
         'filename': 'unique_twitter_words_rank_colored.png', 'title': 'Words only in Twitter'},
        {'name': 'Google', 'column': 'google_rank', 'color': '#4285F4', 
         'filename': 'unique_google_words_rank_colored.png', 'title': 'Words only in Google'},
        {'name': 'NYT', 'column': 'nyt_rank', 'color': '#000000', 
         'filename': 'unique_nyt_words_rank_colored.png', 'title': 'Words only in NYT'},
        {'name': 'Lyrics', 'column': 'lyrics_rank', 'color': '#1DB954', 
         'filename': 'unique_lyrics_words_rank_colored.png', 'title': 'Words only in Lyrics'}
    ]
    
    # Create DataFrame with all needed columns
    plot_df = df[base_cols + rank_cols].copy()
    
    # Drop rows where happiness values are missing
    plot_df = plot_df.dropna(subset=["happiness_average", "happiness_standard_deviation"])
    
    # Convert rank columns to numeric (errors='coerce' turns '--' into NaN)
    for col in rank_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    
    # Count how many corpora each word appears in
    plot_df['n_corpora'] = plot_df[rank_cols].notna().sum(axis=1)


    # Scatter plot: average happiness vs standard deviation -> words in only one corpus and their frequency
    # Create separate plots for each corpus

    for corpus in corpus_info:
        # Words that are ONLY in this corpus (appear in exactly 1 corpus AND that corpus is this one)
        only_in_this = (
            (plot_df['n_corpora'] == 1) & 
            (plot_df[corpus['column']].notna())
        )

        # All words that are in this corpus
        # words_in_corpus = plot_df[corpus['column']].notna()
        
        # Get the unique words for this corpus
        unique_words = plot_df[only_in_this].copy() # change variable name according to previous line
        
        # Create the plot
        plt.figure(figsize=(10, 8))

        # Plot unique words colored by rank (lower rank = darker color)
        if len(unique_words) > 0:
            # For rank: lower number = more frequent = darker color
            # We'll invert the rank so that frequent words (rank 1) get darkest color
            max_rank = 5000  # Since ranks go up to 5000
            
            # Create a color map: darker = more frequent (lower rank)
            
            # Normalize rank to 0-1 range, but invert so frequent = 1, rare = 0:
            # norm_rank = 1 - (unique_words[corpus['column']] / max_rank)
            
            actual_ranks = unique_words[corpus['column']]

            scatter = plt.scatter(
                unique_words["happiness_average"],
                unique_words["happiness_standard_deviation"],
                s=50,
                alpha=1, # fixed alpha for normalised
                # alpha=opacity,
                c=actual_ranks,
                cmap='PuRd_r',  # _r for reversed
                edgecolors='lightgrey',
                linewidth=0.5,
                vmin=1, vmax=5000
            )

            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Frequency rank (low number = more frequent)', fontsize=10)
            
            # Create custom legend entries for rank ranges
            n_unique = len(unique_words)
            
            # Add annotation with rank stats
            median_rank = unique_words[corpus['column']].median()
            min_rank = unique_words[corpus['column']].min()
            max_rank_val = unique_words[corpus['column']].max()

            plt.text(0.02, 0.09, 
                     f'Unique words: {n_unique}\nRank range: {min_rank:.0f} - {max_rank_val:.0f}\nMedian rank: {median_rank:.0f}',
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
            plt.title(f"{corpus['title']}\nColored by frequency rank (darker = more frequent)", 
                 fontsize=14, fontweight='bold')
        else:
            plt.title(f"{corpus['title']}\nNo unique words found", fontsize=14, fontweight='bold')
            plt.text(0.5, 0.5, f"No words unique to {corpus['name']}", 
                 ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.xlabel("Happiness average", fontsize=12)
        plt.ylabel("Happiness standard deviation", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(corpus['filename'])
        plt.close()
        print(f"Saved: {corpus['filename']} with {len(unique_words)} unique words")
# ----- end of code for extra figures -----
        
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
    Corpus comparison:
    - How many words have a rank in each corpus?
    - Simple overlap table (number of words appearing in both corpora)
    - At least one figure about corpus differences (using bar chart here)
    - Provide a concrete example of "common in one corpus but missing in another"
    """
    print_section("2.3 Corpus comparison: corpora and ranks")

    rank_cols = ["twitter_rank", "google_rank", "nyt_rank", "lyrics_rank"]

    # 2.3 (A) 每个语料中，有 rank 的词数（即出现在该语料 top 5000 的词数）
    # 2.3 (A) Number of words with a rank in each corpus (i.e., words appearing in that corpus's top 5000)
    coverage = (
        df[rank_cols]
        .notna()           # True/False：if it has a rank
        .sum()             # count of True values in each column 
        .reset_index()
    )
    coverage.columns = ["rank_column", "n_words_with_rank"]
    coverage["share_of_lexicon"] = coverage["n_words_with_rank"] / len(df)

    print("Words with a non-missing rank in each corpus:")
    print(coverage.to_string(index=False))
    save_csv(coverage, "corpus_rank_coverage.csv", index=False)

    # 覆盖率的柱状图
    # Bar chart (coverage)
    plt.figure()
    plt.bar(coverage["rank_column"], coverage["n_words_with_rank"])
    plt.title("How many labMT words appear in each corpus top-5000?", fontweight='bold')
    plt.xlabel("Corpora")
    plt.ylabel("Number of words with a rank")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    save_figure("bar_corpus_rank_coverage.png")
    plt.close()

    # 2.3 (B) Overlap: 语料之间的共同词
    # 2.3 (B) Overlap patterns: which corpora contain which words?
    # flags: 每一列是 True/False 表示这个词在该语料是否有 rank
    # flags: each column is True/False indicating whether the word has a rank in that corpus
    flags = pd.DataFrame(
        {
            "twitter": df["twitter_rank"].notna(),
            "google": df["google_rank"].notna(),
            "nyt": df["nyt_rank"].notna(),
            "lyrics": df["lyrics_rank"].notna(),
        }
    )

    labels = ["twitter", "google", "nyt", "lyrics"]

    patterns = (
    flags.astype(int)
         .astype(str)
         .agg("".join, axis=1)
    )

    pattern_counts = patterns.value_counts().reset_index()
    pattern_counts.columns = ["corpora_present", "n_words"]

    print("\nMost common overlap patterns (top 12):")
    print(pattern_counts.head(12).to_string(index=False))
    save_csv(pattern_counts, "corpus_overlap_patterns.csv", index=False)

    # 计算每一对语料的“交集大小”：同时为 True 的行数
    # Calculate intersection size for each pair of corpora: count rows where both are True
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

    # 2.3 (C)

    ## Bar chart (word presence by number of corpora) ##

    pattern_counts['n_corpora'] = pattern_counts['corpora_present'].str.count('1')
    by_categories = pattern_counts.groupby('n_corpora')['n_words'].sum()

    plt.figure (figsize=(10, 6))
    y_values = by_categories.values.tolist()
    x_labels = ['0 corpora', '1 corpus', '2 corpora', '3 corpora', '4 corpora']
    colors = ['#999999', '#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    bars = plt.bar(x_labels, y_values, color=colors)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=11)
        
    plt.title('Words by number of corpora they appear in', fontweight='bold')
    plt.xlabel('Number of corpora', fontsize=12)
    plt.ylabel('Number of words', fontsize=12)

    # for i, v in enumerate(total_per_corpus):
    #     plt.text(i, v + 50, str(v), ha='center', fontsize=10)

    plt.tight_layout()
    save_figure("bar_word_presence_by_corpora.png")
    plt.close()

    ## Heatmap (word overlap) ##

    corpora = ['twitter', 'google', 'nyt', 'lyrics']
    n = len(corpora)

    overlap_matrix = pd.DataFrame(0, index=corpora, columns=corpora)

    for _, row in pairwise_overlap.iterrows():
        pair = row['pair']
        count = row['n_words_in_both']

        # Split the pair (e.g., "google+lyrics" -> ["google", "lyrics"])
        c1, c2 = pair.split('+')
        overlap_matrix.loc[c1, c2] = count
        overlap_matrix.loc[c2, c1] = count  # Make symmetric

    # Add diagonal (words in each corpus)
    for corpus in corpora:
        overlap_matrix.loc[corpus, corpus] = flags[corpus].sum()

    # Creak mask for upper triangle
    mask = np.triu(np.ones_like(overlap_matrix, dtype=bool), k=1)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(overlap_matrix, 
                mask=mask,
                annot=True,  # Show values
                fmt=',d',    # Format as integers with commas
                cmap='YlOrRd',  # Color scheme (Yellow-Orange-Red)
                square=True,  # Make cells square
                cbar_kws={'label': 'Number of words'})

    plt.title('Word Overlaps Between Corpora', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure('heatmap_word_overlap.png')
    plt.close()

    # “在 Twitter 中很常见，但在 NYT top-5000 里完全缺失的词”
    # Words that are relatively frequent on Twitter but do NOT appear in NYT's top-5000.
    twitter_common_nyt_missing = (
        df[(df["twitter_rank"].notna()) & (df["nyt_rank"].isna())]
        .sort_values("twitter_rank")  # twitter_rank 越小越常见
        .head(20)[["word", "twitter_rank", "happiness_average"]]
    )

    print("\nExample: words frequent on Twitter but missing in NYT top-5000 (top 20):")
    print(twitter_common_nyt_missing.to_string(index=False))
    save_csv(twitter_common_nyt_missing, "twitter_common_nyt_missing_top20.csv", index=False)

    # Optional: compare ranks directly for words present in BOTH corpora.
    # (This can hint at how similar/different the corpora are.)

    both_twitter_nyt = df.dropna(subset=["twitter_rank", "nyt_rank"])

    plt.figure(
        # figsize=(10, 8)
        )

    scatter_twt_nyt = plt.scatter(both_twitter_nyt["twitter_rank"], 
                both_twitter_nyt["nyt_rank"], 
                c=both_twitter_nyt["happiness_average"], #colour by happiness rank
                s=20, 
                alpha=0.8,
                cmap='BrBG',
                edgecolors='face',
                # linewidth=0.3,
                vmin=1, vmax=9
                )
    # Add colorbar
    cbar = plt.colorbar(scatter_twt_nyt)
    cbar.set_label('Happiness average (1-9)', fontsize=10)

    plt.title("Twitter rank vs NYT rank (words present in both)", fontweight='bold')
    plt.xlabel("Twitter rank (1 = most frequent)")
    plt.ylabel("NYT rank (1 = most frequent)")
    plt.tight_layout()
    save_figure("scatter_twitter_rank_vs_nyt_rank.png")
    plt.close()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    # 先复用之前写好的加载 + 清洗函数
    # First, reuse the previously written loading and cleaning functions
    df = load_labmt()

    # 2.1 Distribution
    analyse_happiness_distribution(df)

    # 2.2 Disagreement
    analyse_disagreement(df)

    # 2.3 Corpus comparison
    analyse_corpora(df)


if __name__ == "__main__":
    main()