"""Seminar 3 — Hedonometer / labMT 1.0 demo analysis

This script is intentionally written in a *step-by-step* style (more like a
notebook than a software package). The goal is to make each move legible:

- load a real dataset
- look at its structure and missing values
- run a few sanity checks
- make a handful of simple plots and tables

It follows the same sequence as the Seminar 3 assignment tasks.

Run (from the project root):
    python src/hedonometer_labmt_demo.py

What you get after running:
- figures/  (PNG plots)
- tables/   (CSV tables you can open in Excel/Numbers)

Dataset:
- data/raw/Data_Set_S1.txt
  (tab-delimited; the first 3 lines are metadata, then the header row)

Dependencies (see requirements.txt):
- pandas
- numpy
- matplotlib
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
# These are the only functions in this file. Everything else runs sequentially.
# We use helpers only for repeated patterns (printing sections, saving outputs).


def print_section(title: str) -> None:
    """Print a clear section divider in the terminal."""
    bar = "=" * 90
    print("\n" + bar)
    print(title)
    print(bar)


def save_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """Save a DataFrame to tables/ and print where it went."""
    out_path = TABLES_DIR / filename
    df.to_csv(out_path, index=index)
    print(f"Saved table: {out_path}")


def save_figure(filename: str, dpi: int = 200) -> None:
    """Save the current matplotlib figure to figures/ and print where it went."""
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=dpi)
    print(f"Saved figure: {out_path}")


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
# We build file paths relative to THIS script, so the code works on any machine
# as long as the folder structure is the same.

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Data_Set_S1.txt"

FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# 1. LOAD, CLEAN, AND DESCRIBE THE DATASET
# -----------------------------------------------------------------------------

print_section("1.1 Load the dataset (Data_Set_S1.txt)")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Expected to find: "
        f"{DATA_PATH}\n\n"
        "Make sure Data_Set_S1.txt is in data/raw/ and try again."
    )

# The file begins with 3 lines of metadata, then a header row.
# The rank columns use '--' to mean "this word is NOT in that corpus's top-5000".
# We convert '--' to a proper missing value (NaN) so pandas can work with it.

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    skiprows=3,
    na_values=["--"],
    encoding="utf-8",
)

# Convert numeric columns to numeric dtypes.
# (When a column has missing values, pandas often uses floats to allow NaN.)

numeric_cols = [
    "happiness_rank",
    "happiness_average",
    "happiness_standard_deviation",
    "twitter_rank",
    "google_rank",
    "nyt_rank",
    "lyrics_rank",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["word"] = df["word"].astype("string")

print("First 8 rows:")
print(df.head(8))
print("\nShape (rows, columns):", df.shape)

# A small preview table can be useful for quick inspection.
save_csv(df.head(50), "preview_first_50_rows.csv", index=False)


print_section("1.2 Data dictionary + missing values")

# A "data dictionary" is a plain-language explanation of each column.
# Here we create a table with:
# - column name
# - pandas dtype
# - number of missing values
# - one example value

col_dtypes = df.dtypes.astype(str).reset_index()
col_dtypes.columns = ["column", "dtype"]

missing = df.isna().sum().reset_index()
missing.columns = ["column", "n_missing"]

example_values = df.iloc[0].reset_index()
example_values.columns = ["column", "example_value"]

data_dictionary = col_dtypes.merge(missing, on="column").merge(example_values, on="column")

print(data_dictionary.to_string(index=False))
save_csv(data_dictionary, "data_dictionary.csv", index=False)

print("\nReminder about missing ranks:")
print("- If twitter_rank is NaN for a word, that means the word is NOT in Twitter's top-5000 list")
print("  used to build this dataset (not that the word never appears in Twitter).")


print_section("1.3 Sanity checks")

# (A) Duplicated words
n_duplicates = int(df["word"].duplicated().sum())
print("Duplicated words:", n_duplicates)

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
plt.hist(h, bins=40)
plt.title("Distribution of happiness_average (labMT 1.0)")
plt.xlabel("happiness_average (1–9)")
plt.ylabel("number of words")
plt.tight_layout()
save_figure("happiness_average_hist.png")
plt.close()


print_section("2.2 Disagreement: happiness_standard_deviation")

# Scatter: happiness score vs standard deviation
plt.figure()
plt.scatter(
    df["happiness_average"],
    df["happiness_standard_deviation"],
    s=10,
    alpha=0.35,
)
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
coverage_rows = []
for col in rank_cols:
    n_present = int(df[col].notna().sum())
    coverage_rows.append(
        {
            "rank_column": col,
            "n_words_with_rank": n_present,
            "share_of_lexicon": n_present / len(df),
        }
    )

coverage = pd.DataFrame(coverage_rows)
print(coverage.to_string(index=False))
save_csv(coverage, "corpus_rank_coverage.csv", index=False)

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
patterns = flags.apply(lambda row: "+".join([lab for lab in labels if row[lab]]) or "none", axis=1)

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

# 5 very positive
pos5 = df.sort_values("happiness_average", ascending=False).head(5).copy()
pos5["category"] = "very positive"

# 5 very negative
neg5 = df.sort_values("happiness_average", ascending=True).head(5).copy()
neg5["category"] = "very negative"

# 5 highly contested (high standard deviation)
con5 = df.sort_values("happiness_standard_deviation", ascending=False).head(5).copy()
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
