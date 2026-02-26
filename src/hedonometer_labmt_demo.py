from __future__ import annotations

from pathlib import Path
import pandas as pd

# === 路径设置：根据当前脚本的位置，自动找到项目根目录 ===
# 当前文件在 hedonometer-project-group6/src/load_labmt.py
# parents[0] -> src
# parents[1] -> hedonometer-project-group6  (项目根目录)
BASE_DIR = Path(__file__).resolve().parents[1]

# 数据文件就在 项目根目录/data/raw/Data_Set_S1.txt
RAW_PATH = BASE_DIR / "data" / "raw" / "Data_Set_S1.txt"


def load_labmt(path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Load labMT 1.0 tab-delimited file.

    Assumptions from assignment:
    - File may contain metadata/comment lines before the header.
    - Missing ranks are represented as '--'.
    """


    # 读取整个文件，找到真正的表头行
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # 启发式：表头那一行里既有 'word' 又有 'happiness'
    header_idx: int | None = None
    for i, line in enumerate(lines[:200]):  # header 应该在文件前 200 行之内
        if "\t" in line and "word" in line.lower() and "happiness" in line.lower():
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            "Could not find header row. Inspect the raw file and adjust header detection."
        )

    # 从表头开始，用 pandas 读入
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=header_idx,
        na_values=["--"],        # 把 '--' 当作缺失值
        keep_default_na=True,
        dtype=str,               # 先按字符串读，后面再转数值
    )

    # 去掉列名两端的空格
    df.columns = [c.strip() for c in df.columns]

<<<<<<< HEAD
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    skiprows=3,
    na_values=["--"],
    encoding="utf-8",
    dtype={"word": "string"}
)

# Convert numeric columns to numeric dtypes.
# (When a column has missing values, pandas often uses floats to allow NaN.)
# Explicitly set the 'word' column as string to avoid implicit object dtype inference and ensure a clear, reproducible schema.

numeric_cols = [
    "happiness_rank",
    "happiness_average",
    "happiness_standard_deviation",
    "twitter_rank",
    "google_rank",
    "nyt_rank",
    "lyrics_rank",
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Convert all numeric columns together to enforce a consistent numeric schema.

df["word"] = df["word"].astype("string")

print("First 8 rows:")
print(df.head(8))
print("\nShape (rows, columns):", df.shape)

# A small preview table can be useful for quick inspection.
save_csv(df.head(50), "preview_first_50_rows.csv", index=False)
=======
    # 除了 'word' 以外的列都尝试转成数值
    numeric_cols = [c for c in df.columns if c != "word"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 处理 word 列：转成字符串并去掉空格
    df["word"] = df["word"].astype(str).str.strip()

    return df
>>>>>>> 377daced21773e90347cb998f7f89500bd341f4e


def main() -> None:
    df = load_labmt()

    # === 1.1 + 1.2 + 1.3 所有需要的输出 ===

    # 1) 形状（行 × 列）
    print("Shape (rows, cols):", df.shape)

    # 2) 列名列表
    print("\nColumns:", list(df.columns))

    # 3) 每一列的数据类型（data dictionary 用）
    print("\nData types:\n", df.dtypes)

<<<<<<< HEAD
data_dictionary = col_dtypes.merge(missing, on="column").merge(example_values, on="column")

print(data_dictionary.to_string(index=False))
save_csv(data_dictionary, "data_dictionary.csv", index=False)

print("\nReminder about missing ranks:")
print("- If twitter_rank is NaN for a word, that means the word is NOT in Twitter's top-5000 list")
print("  used to build this dataset (not that the word never appears in Twitter).")


print_section("1.3 Sanity checks")

# (A) Duplicated words
duplicates = df[df["word"].duplicated()]
n_duplicates = len(duplicates)
if not duplicates.empty:
    print("Duplicate words found:")
    print(duplicates)
print("Duplicated words:", n_duplicates)

# If duplicate words exist, print them for inspection rather than only counting them.

# (B) A reproducible random sample
sample_15 = df.sample(15, random_state=42)
print("\nRandom sample (15 rows):")
print(sample_15)
save_csv(sample_15, "random_sample_15_rows.csv", index=False)

# (C) Top positive / negative words by happiness_average
show_cols = ["word", "happiness_average", "happiness_standard_deviation"]

top_10_positive = df.sort_values("happiness_average", ascending=False).head(10)[show_cols]
top_10_negative = df.sort_values("happiness_average", ascending=True).head(10)[show_cols]

print("\nTop 10 positive words (by happiness_average):")
print(top_10_positive.to_string(index=False))
print("\nTop 10 negative words (by happiness_average):")
print(top_10_negative.to_string(index=False))

save_csv(top_10_positive, "top_10_positive_words.csv", index=False)
save_csv(top_10_negative, "top_10_negative_words.csv", index=False)


# -----------------------------------------------------------------------------
# 2. QUANTITATIVE EXPLORATION
# -----------------------------------------------------------------------------

print_section("2.1 Distribution of happiness_average")

h = df["happiness_average"].dropna()
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

# Histogram
plt.figure()
plt.hist(h, bins=40, edgecolor="black")
plt.title("Distribution of happiness_average (labMT 1.0)")
plt.xlabel("happiness_average (1–9)")
plt.ylabel("number of words")
plt.tight_layout()
save_figure("happiness_average_hist.png")
plt.close()


print_section("2.2 Disagreement: happiness_standard_deviation")

# Scatter: happiness score vs standard deviation
plt.figure()

scatter_df = df[["happiness_average", "happiness_standard_deviation"]].dropna()

plt.scatter(
    scatter_df["happiness_average"],
    scatter_df["happiness_standard_deviation"],
    s=10,
    alpha=0.35,
)

# Use only complete cases (drop rows with missing values) before plotting.

plt.title("Disagreement vs score: happiness_average vs happiness_standard_deviation")
plt.xlabel("happiness_average")
plt.ylabel("happiness_standard_deviation")
plt.tight_layout()
save_figure("happiness_vs_std_scatter.png")
plt.close()

# Which words do people disagree about most?
most_contested_15 = df.sort_values("happiness_standard_deviation", ascending=False).head(15)[show_cols]
print("Top 15 most 'contested' words (highest standard deviation):")
print(most_contested_15.to_string(index=False))
save_csv(most_contested_15, "top_15_contested_words.csv", index=False)


print_section("2.3 Corpus comparison: rank coverage + overlaps")

rank_cols = ["twitter_rank", "google_rank", "nyt_rank", "lyrics_rank"]

# (A) Coverage: how many words have a rank in each corpus?
coverage = (
    df[rank_cols]
    .notna()
    .sum()
    .reset_index()
)

coverage.columns = ["rank_column", "n_words_with_rank"]
coverage["share_of_lexicon"] = coverage["n_words_with_rank"] / len(df)

print(coverage.to_string(index=False))
save_csv(coverage, "corpus_rank_coverage.csv", index=False)

# Replace the loop with a vectorized pandas operation to compute rank coverage more clearly and efficiently.

# Bar chart (coverage)
plt.figure()
plt.bar(coverage["rank_column"], coverage["n_words_with_rank"])
plt.title("How many labMT words appear in each corpus top-5000?")
plt.xlabel("corpus rank column")
plt.ylabel("number of words with a rank")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
save_figure("corpus_rank_coverage_bar.png")
plt.close()

# (B) Overlap patterns: which corpora contain which words?
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

# A small table of pairwise overlaps can be easier to discuss in writing.
pairs = []
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        a, b = labels[i], labels[j]
        pairs.append({"pair": f"{a}+{b}", "n_words_in_both": int((flags[a] & flags[b]).sum())})

pairwise_overlap = pd.DataFrame(pairs).sort_values("pair")
save_csv(pairwise_overlap, "pairwise_overlap_counts.csv", index=False)

# (C) One concrete example: frequent in one corpus, missing in another.
# Here we look for words that are relatively frequent on Twitter but do NOT appear in NYT's top-5000.

twitter_common_nyt_missing = (
    df[(df["twitter_rank"].notna()) & (df["nyt_rank"].isna())]
    .sort_values("twitter_rank")
    .head(20)[["word", "twitter_rank", "happiness_average"]]
)

print("\nExample words frequent on Twitter but missing in NYT top-5000 (top 20 by twitter_rank):")
print(twitter_common_nyt_missing.to_string(index=False))
save_csv(twitter_common_nyt_missing, "twitter_common_nyt_missing_top20.csv", index=False)

# Optional: compare ranks directly for words present in BOTH corpora.
# (This can hint at how similar/different the corpora are.)

both_twitter_nyt = df.dropna(subset=["twitter_rank", "nyt_rank"])

plt.figure()
plt.scatter(both_twitter_nyt["twitter_rank"], both_twitter_nyt["nyt_rank"], s=10, alpha=0.35)
plt.title("Twitter rank vs NYT rank (words present in both)")
plt.xlabel("twitter_rank (1 = most frequent)")
plt.ylabel("nyt_rank (1 = most frequent)")
plt.tight_layout()
save_figure("twitter_rank_vs_nyt_rank_scatter.png")
plt.close()


# -----------------------------------------------------------------------------
# 3. QUALITATIVE EXPLORATION (A SMALL 'EXHIBIT' OF WORDS)
# -----------------------------------------------------------------------------

print_section("3.1 Word exhibit (one possible way to select 20 words)")

# In your project, you should *choose* your words (and justify your choices).
# Here we generate a starter exhibit automatically to show how you might do it.

# Helper function to avoid repeating sorting logic
def top_n(df, column, n=5, ascending=False):
    """
    Return the top n rows sorted by a given column.
    """
    return df.sort_values(column, ascending=ascending).head(n).copy()

# 5 very positive
pos5 = top_n(df, "happiness_average", n=5, ascending=False)
pos5["category"] = "very positive"

# 5 very negative
neg5 = top_n(df, "happiness_average", n=5, ascending=True)
neg5["category"] = "very negative"

# 5 highly contested (high standard deviation)
con5 = top_n(df, "happiness_standard_deviation", n=5, ascending=False)
con5["category"] = "highly contested"

# 5 platform-specific (frequent in Twitter, missing in NYT)
# We take words with small twitter_rank (more frequent) that have no nyt_rank.
plat5 = (
    df[(df["twitter_rank"].notna()) & (df["nyt_rank"].isna())]
    .sort_values("twitter_rank")
    .head(5)
    .copy()
)
plat5["category"] = "Twitter-common, NYT-missing"

exhibit_cols = [
    "category",
    "word",
    "happiness_average",
    "happiness_standard_deviation",
    "twitter_rank",
    "google_rank",
    "nyt_rank",
    "lyrics_rank",
]

exhibit = pd.concat([pos5, neg5, con5, plat5], ignore_index=True)[exhibit_cols]

print(exhibit.to_string(index=False))
save_csv(exhibit, "word_exhibit_demo_20_words.csv", index=False)


# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------

print_section("Done")
print("If you embed figures in a README, use relative paths like:  ![](figures/your_plot.png)")
print("Figures folder:", FIGURES_DIR)
print("Tables folder:", TABLES_DIR)
=======
    # 4) 每列缺失值数量
    print(
        "\nMissing values per column:\n",
        df.isna().sum().sort_values(ascending=False),
    )

    # 5) 单词是否有重复
    print("\nDuplicate words:", df["word"].duplicated().sum())

    # 6) 随机抽样 15 行
    print("\nSample rows:\n", df.sample(15, random_state=42))


if __name__ == "__main__":
    main()
>>>>>>> 377daced21773e90347cb998f7f89500bd341f4e
