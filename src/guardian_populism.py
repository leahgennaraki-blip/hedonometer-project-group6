import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


# ---------- SETTINGS: ADAPT THESE ----------
DATA_FILE   = "data/processed/guardian_articles_with_scores.csv"
DATE_COL    = "pub_date"
PERIOD_COL    = "period"
SECTION_COL = "section_name"
TEXT_COL    = "body_text"
HAPPY_COL   = "happiness"
# ------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = ROOT / "data" / "raw"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"


DATA_RAW_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


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

populism_path = f"{DATA_RAW_DIR}/measuring_populism_dict.csv"
df_populism_terms = pd.read_csv(populism_path, usecols=["UK"])
print(df_populism_terms)

# -----------------------------------------------------

pattern = r"\b(" + "|".join(re.escape(t) for t in df_populism_terms) + r")\b"

# Count term hits per article
df["populism_count"] = df[TEXT_COL].str.count(pattern, flags=re.IGNORECASE)

# Normalized frequency: far-right terms per 1,000 words
df["populism_per_1000"] = (df["populism_count"] / df["total_words"]) * 1000

print(df[[SECTION_COL, PERIOD_COL, "year", "populism_count", "populism_per_1000", HAPPY_COL]])

# Articles that contain at least one far-right term
df_populism = df[df["populism_count"] > 0].copy()

# -------- Overall summary for these articles --------
total_articles_far = len(df_populism)
total_populism_terms    = df_populism["populism_count"].sum()
total_words_populism   = df_populism["total_words"].sum()
overall_populism_per_1000 = (total_populism_terms / total_words_populism) * 1000
mean_happiness_populism   = df_populism[HAPPY_COL].mean()

print("Articles with >= 1 far-right term:")
print("  Number of articles:", total_articles_far)
print("  Total populist term occurrences:", total_populism_terms)
print("  Total words (in these articles):", total_words_populism)
print("  Far-right terms per 1,000 words (overall):", overall_populism_per_1000)
print("  Mean happiness score (these articles):", mean_happiness_populism)

# -------- Optional: by period and section --------
summary_populism_by_group = (
    df_populism
    .groupby([PERIOD_COL, SECTION_COL])
    .agg(
        n_articles      = ("populism_count", "size"),
        total_populism = ("populism_count", "sum"),
        total_words     = ("total_words", "sum"),
        mean_happiness  = (HAPPY_COL, "mean"),
    )
    .reset_index()
)

summary_populism_by_group["populism_per_1000"] = (
    summary_populism_by_group["total_populism"] /
    summary_populism_by_group["total_words"] * 1000
)

print("\nPopulist term frequency by period and section (only articles with >= 1 far-right term):\n")
print(summary_populism_by_group)

# Summary over ALL articles in your sample
summary_all = (
    df
    .groupby([PERIOD_COL, SECTION_COL])
    .agg(
        n_articles            = (TEXT_COL, "size"),                 # all articles in group
        total_words           = ("total_words", "sum"),             # all words in group
        total_populism        = ("populism_count", "sum"),          # all term hits (including zeros)
        n_populist_articles   = ("populism_count", lambda x: (x > 0).sum()),  # articles with ≥1 term
    )
    .reset_index()
)

# 1) Term frequency per 1,000 words in the whole sample
summary_all["populism_terms_per_1000_words"] = (
    summary_all["total_populism"] / summary_all["total_words"] * 1000
)

# 2) Share of articles with ≥1 populist term
summary_all["share_articles_with_populism"] = (
    summary_all["n_populist_articles"] / summary_all["n_articles"]
)

print("Populist term frequency in the whole sample:\n")
print(summary_all)


# ---------- CLEANER PLOTS: POINT + LINE CHARTS ----------

sns.set_theme(style="whitegrid")

section_order = ["Opinion", "Politics", "World news"]
period_order = ["2010-2013", "2020-2023"]

summary_all["section_name"] = pd.Categorical(
    summary_all["section_name"],
    categories=section_order,
    ordered=True
)
summary_all["period"] = pd.Categorical(
    summary_all["period"],
    categories=period_order,
    ordered=True
)

summary_populism_by_group["section_name"] = pd.Categorical(
    summary_populism_by_group["section_name"],
    categories=section_order,
    ordered=True
)
summary_populism_by_group["period"] = pd.Categorical(
    summary_populism_by_group["period"],
    categories=period_order,
    ordered=True
)

# ---------- FIGURE 1: WHOLE SAMPLE (INTENSITY + PREVALENCE) ----------
fig1, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

# Panel A: intensity in whole sample
ax1 = axes[0]
sns.pointplot(
    data=summary_all,
    x="period",
    y="populism_terms_per_1000_words",
    hue="section_name",
    ax=ax1,
    errorbar=None,
    markers="o",
    linestyles="-"
)
ax1.set_ylabel("Populist terms per 1,000 words")
ax1.set_xlabel("")
ax1.set_title("Intensity of populist terms (all articles)")
ax1.legend(title="Section", loc="upper left")

# Panel B: prevalence in whole sample
ax2 = axes[1]
sns.pointplot(
    data=summary_all,
    x="period",
    y="share_articles_with_populism",
    hue="section_name",
    ax=ax2,
    errorbar=None,
    markers="o",
    linestyles="-"
)
ax2.set_ylabel("Share of articles with ≥1 populist term")
ax2.set_xlabel("")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.set_title("Prevalence of populist terms (all articles)")
# Avoid duplicate legend on second panel
ax2.get_legend().remove()

plt.tight_layout()
fig1.savefig(FIGURES_DIR / "fig_populism_whole_sample_pointline.png", dpi=300)
plt.close(fig1)

print("Saved Figure 1 (point+line) to",
    FIGURES_DIR / "fig_populism_whole_sample_pointline.png")

# ---------- FIGURE 2: ONLY ARTICLES WITH ≥1 POPULIST TERM ----------
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

# Panel A: intensity among populist articles only
ax3 = axes2[0]
sns.pointplot(
    data=summary_populism_by_group,
    x="period",
    y="populism_per_1000",
    hue="section_name",
    ax=ax3,
    errorbar=None,
    markers="o",
    linestyles="-"
)
ax3.set_ylabel("Populist terms per 1,000 words\n(articles with ≥1 term)")
ax3.set_xlabel("")
ax3.set_title("Intensity (only populist articles)")
ax3.legend(title="Section", loc="upper left")

# Panel B: mean happiness among populist articles
ax4 = axes2[1]
sns.pointplot(
    data=summary_populism_by_group,
    x="period",
    y="mean_happiness",
    hue="section_name",
    ax=ax4,
    errorbar=None,
    markers="o",
    linestyles="-"
)
ax4.set_ylabel("Mean happiness score\n(articles with ≥1 term)")
ax4.set_xlabel("")
ax4.set_title("Happiness (only populist articles)")
ax4.get_legend().remove()

plt.tight_layout()
fig2.savefig(FIGURES_DIR / "fig_populism_populist_articles_pointline.png", dpi=300)
plt.close(fig2)

print("Saved Figure 2 (point+line) to",
    FIGURES_DIR / "fig_populism_populist_articles_pointline.png")

