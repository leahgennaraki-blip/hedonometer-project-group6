import pandas as pd
import numpy as np
import re

# ---------- SETTINGS: ADAPT THESE ----------
DATA_FILE   = "data/processed/guardian_articles_with_scores.csv"
DATE_COL    = "pub_date"
PERIOD_COL    = "period"
SECTION_COL = "section_name"
TEXT_COL    = "body_text"
HAPPY_COL   = "happiness"
# ------------------------------------------

df = pd.read_csv(DATA_FILE)

# Parse date, extract year (optional but useful for "over time")
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])
df["year"] = df[DATE_COL].dt.year

sections_of_interest = ["Politics", "World news", "Opinion"]

df = df[df[SECTION_COL].isin(sections_of_interest)]

# We DO NOT create period; we use your existing column
# Just check some rows to ensure period is as expected
print(df[[SECTION_COL, PERIOD_COL, "year", HAPPY_COL]])

# Basic text cleaning and word count
df[TEXT_COL] = df[TEXT_COL].fillna("").str.lower()
df["total_words"] = df[TEXT_COL].str.split().str.len()
df = df[df["total_words"] > 0]

print(df[[SECTION_COL, PERIOD_COL, "year", HAPPY_COL, "total_words"]])


# ---------- FAR-RIGHT DICTIONARY: EDIT THIS ----------
far_right_terms = [
    "immigration", "nation", "immigrants", "immigrant",
    "illegal alien", "illegal aliens", "illegal immigrants", "illegal immigrant",
    "national pride", "Make America Great Again", "MAGA", "corrupt", "undemocratic"
]
# -----------------------------------------------------

pattern = r"\b(" + "|".join(re.escape(t) for t in far_right_terms) + r")\b"

# Count term hits per article
df["far_right_count"] = df[TEXT_COL].str.count(pattern, flags=re.IGNORECASE)

# Normalized frequency: far-right terms per 1,000 words
df["far_right_per_1000"] = (df["far_right_count"] / df["total_words"]) * 1000

print(df[[SECTION_COL, PERIOD_COL, "year", "far_right_count", "far_right_per_1000", HAPPY_COL]])

# Articles that contain at least one far-right term
df_far_right = df[df["far_right_count"] > 0].copy()

# -------- Overall summary for these articles --------
total_articles_far = len(df_far_right)
total_far_right_terms    = df_far_right["far_right_count"].sum()
total_words_far    = df_far_right["total_words"].sum()
overall_far_per_1000 = (total_far_right_terms / total_words_far) * 1000
mean_happiness_far   = df_far_right[HAPPY_COL].mean()

print("Articles with >= 1 far-right term:")
print("  Number of articles:", total_articles_far)
print("  Total far-right term occurrences:", total_far_right_terms)
print("  Total words (in these articles):", total_words_far)
print("  Far-right terms per 1,000 words (overall):", overall_far_per_1000)
print("  Mean happiness score (these articles):", mean_happiness_far)

# -------- Optional: by period and section --------
summary_far_by_group = (
    df_far_right
    .groupby([PERIOD_COL, SECTION_COL])
    .agg(
        n_articles      = ("far_right_count", "size"),
        total_far_right = ("far_right_count", "sum"),
        total_words     = ("total_words", "sum"),
        mean_happiness  = (HAPPY_COL, "mean"),
    )
    .reset_index()
)

summary_far_by_group["far_right_per_1000"] = (
    summary_far_by_group["total_far_right"] /
    summary_far_by_group["total_words"] * 1000
)

print("\nBy period and section (only articles with >= 1 far-right term):")
print(summary_far_by_group)