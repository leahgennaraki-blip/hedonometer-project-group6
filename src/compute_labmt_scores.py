# src/compute_labmt_scores.py
from pathlib import Path
import re

import numpy as np
import pandas as pd

# For stopwords
# import nltk
# from nltk.corpus import stopwords


from load_labmt import load_labmt  # 按你自己的实现保持不变 # Remain unchanged according to your own implementation


# ----------------------------------------------------------------------
# METHODOLOGICAL CHOICES
# ----------------------------------------------------------------------
# 1. Neutral words:
# 
# method 1 (default): Remove all words with scores between 4 and 6 (neutral range).
#   - Keeps only emotionally charged words (<4 or >6).
#   - Simplest and most aggressive.
#
# method 2: Remove neutral words that have low standard deviation.
#   - Keeps neutral words with high variance (ambiguous usage).
#   - Requires standard deviation column in labMT file and uses STD_THRESHOLD.
#   - We remove words that are BOTH emotionally neutral (score between 4 and 6)
#    AND have low threshold (i.e., are consistently rated neutral). Neutral words 
#    with high std are kept because they may carry emotional nuances in context. 
#    Words outside the neutral range are kept regardless.
#
# method 3: Remove neutral words that are also English stopwords.
#   - Keeps neutral content words (e.g., "government") but removes neutral
#     function words (e.g., "the", "and").
#   - Requires NLTK stopwords.
#   - We remove English stopwords (e.g., "the", "and", "of"). Neutral content words
#    (like "government", "economy") are kept because they may carry topical
#    relevance. This balances sensitivity and coverage.
# 
#
# 2. Words not found in the lexicon: ignored; we report coverage.
#
# 3. Preprocessing: lowercasing, alphabetic tokens only.
#    - Lowercase all text.
#    - Extract alphabetic sequences, allowing apostrophes inside words
#      (e.g., "someone's" is kept as one token). But other punctuations
#      are removed.
#    - Hyphenated words become separate tokens (hyphens are not kept)
# ----------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z']+")


# ---------- ADJUST THESE CONSTANTS ----------
NEUTRAL_MIN = 4.0
NEUTRAL_MAX = 6.0
# ---------------------------------------------

# # Standard deviation threshold. If None, the median of all std devs is used.
# STD_THRESHOLD = 2   # you can set a number like 1.2, or leave as None to use median
# # ---------------------------------------------


def tokenize(text: str):
    """把文章文本转成小写单词列表。
    Convert the article text into a list of lowercase words."""
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())

# ===================== METHOD 1: Remove all neutral words =====================
# 4-6 FILTER method
# NEW: Simple filter that removes all neutral words (no std dev used)
def filter_neutral_all(word_to_score: dict) -> dict:
    """
    Keep only words with score < NEUTRAL_MIN or > NEUTRAL_MAX.
    Remove all words in the neutral range.
    """
    filtered = {}
    for word, score in word_to_score.items():
        if score < NEUTRAL_MIN or score > NEUTRAL_MAX:
            filtered[word] = score
    return filtered

# ===================== METHOD 2: Neutral + low std dev =====================
# STD method
# NEW: Function to compute median standard deviation from the full lexicon
# def compute_median_std(word_to_std: dict) -> float:
#     """Return the median of all standard deviations."""
#     return np.median(list(word_to_std.values()))

# NEW: Filter based on score and std dev only (no stopwords)
# def filter_neutral_low_std(word_to_score_std: dict, std_threshold: float) -> dict:
#     """
#     Apply the refined rule:
#     Keep a word if:
#       - its score is NOT in the neutral range [NEUTRAL_MIN, NEUTRAL_MAX] (emotionally charged), OR
#       - its score IS neutral BUT its standard deviation > std_threshold.
#     In other words, remove only words that are neutral AND have std dev <= std_threshold.
#     """
#     filtered = {}
#     for word, (score, std) in word_to_score_std.items():
#         if score < NEUTRAL_MIN or score > NEUTRAL_MAX:
#             # Emotionally charged – keep regardless
#             filtered[word] = score
#         else:
#             # Neutral score – keep only if std dev is above threshold
#             if std > std_threshold:
#                 filtered[word] = score
    return filtered

# ===================== METHOD 3: Neutral + stopwords =====================
# STOPWORD method
# NEW: Function to load stopwords (entirely new)
# def load_stopwords():
#     """Return a set of English stopwords (lowercase)."""
#     try:
#         stop = set(stopwords.words('english'))
#     except LookupError:
#         nltk.download('stopwords')
#         stop = set(stopwords.words('english'))
#     return stop

# NEW: Function to apply the neutral+stopword filter
# def filter_neutral_stopwords(word_to_score: dict, stopwords_set: set) -> dict:
#     """
#     Apply the methodological rule:
#     Keep a word if:
#       - its score is NOT in the neutral range [NEUTRAL_MIN, NEUTRAL_MAX]
#         (i.e., it is emotionally charged), OR
#       - its score IS neutral BUT it is NOT a stopword.
#     In other words, remove only words that are both neutral AND stopwords.
#     """
#     filtered = {}
#     for word, score in word_to_score.items():
#         if score < NEUTRAL_MIN or score > NEUTRAL_MAX:
#             # Emotionally charged - keep regardless of stopword status
#             filtered[word] = score
#         else:
#             # Neutral score - keep only if its NOT a stopword
#             if word not in stopwords_set:
#                 filtered[word] = score
#     return filtered


def main():
    # 1. 读入 process 后的文章数据
    # 1. Read the article data after processing.
    processed_path = Path("data/processed/guardian_articles_processed.csv")
    df = pd.read_csv(processed_path)
    print("Loaded processed articles:", df.shape)

    # 2. 加载 labMT 词典
    # 2. Load the labMT dictionary
    labmt_path = Path("data/raw/Data_Set_S1.txt")
    labmt_df = load_labmt(labmt_path)

    # === 根据你实际的列名修改下面两个变量 ===
    # === Modify the following two variables according to your actual column names ===
    word_col = "word"               # 单词列 Word list
    score_col = "happiness_average" # happiness 列
    # std_col = "happiness_standard_deviation"   # NEW: standard deviation column
    # =======================================

    # ===================== METHOD 1 (active) =====================
    # CHANGED: Store full lexicon separately before filtering
    full_word_to_score = dict(zip(labmt_df[word_col], labmt_df[score_col]))
    print(f"Loaded labMT lexicon with {len(full_word_to_score)} words")

    # 4-6 FILTER method
    # NEW: Apply the simple neutral-word filter
    word_to_score = filter_neutral_all(full_word_to_score)
    print(f"After removing all neutral words (scores {NEUTRAL_MIN}–{NEUTRAL_MAX}): {len(word_to_score)} words remain")

    # ===================== METHOD 2 (commented out) =====================
    # STD method
    # NEW: Create dictionary mapping word to (score, std)
    # word_to_score_std = dict(zip(labmt_df[word_col], 
    #                               zip(labmt_df[score_col], labmt_df[std_col])))
    # print(f"Loaded labMT lexicon with {len(word_to_score_std)} words")

    # # NEW: Determine std threshold (if None, use median)
    # if STD_THRESHOLD is None:
    #     # Extract all std devs
    #     all_stds = [std for (_, std) in word_to_score_std.values()]
    #     std_threshold = np.median(all_stds)
    #     print(f"Using median standard deviation as threshold: {std_threshold:.3f}")
    # else:
    #     std_threshold = STD_THRESHOLD
    #     print(f"Using user-defined standard deviation threshold: {std_threshold:.3f}")

    # # NEW: Apply the filter
    # word_to_score = filter_neutral_low_std(word_to_score_std, std_threshold)
    # print(f"After removing neutral words with low std dev (scores {NEUTRAL_MIN}–{NEUTRAL_MAX}, std <= {std_threshold:.3f}): {len(word_to_score)} words remain")

    # ===================== METHOD 3 (commented out) =====================
    # STOPWORD method
    # NEW: Load stopwords
    # stop = load_stopwords()
    # print(f"Loaded {len(stop)} English stopwords")

    # NEW: Apply the filter to get the final lexicon
    # word_to_score = filter_neutral_stopwords(full_word_to_score, stop)
    # print(f"After removing neutral stopwords (scores {NEUTRAL_MIN}–{NEUTRAL_MAX} AND stopword): {len(word_to_score)} words remain")

    # 3. 对每篇文章计算 happiness + 命中词数
    # 3. Calculate the happiness + number of hit words for each article.
    def compute_happiness(text: str):
        tokens = tokenize(text)
        scores = [word_to_score[w] for w in tokens if w in word_to_score]
        if not scores:
            return np.nan
        return float(np.mean(scores))

    def count_labmt_words(text: str):
        tokens = tokenize(text)
        return sum(1 for w in tokens if w in word_to_score)
    
    # NEW: Function to count total tokens in an article
    def count_total_tokens(text: str):
        return len(tokenize(text))

    df["happiness"] = df["body_text"].apply(compute_happiness)
    df["labmt_token_count"] = df["body_text"].apply(count_labmt_words)

    # NEW: Add total tokens and coverage proportion
    df["total_tokens"] = df["body_text"].apply(count_total_tokens)
    df["labmt_proportion"] = df["labmt_token_count"] / df["total_tokens"]

    # 4. 保存「逐文章」级别的数据（已经有）
    # 4. Save data at the "per-article" level (already exists)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    scores_path = processed_dir / "guardian_articles_with_scores.csv"
    df.to_csv(scores_path, index=False, encoding="utf-8")
    print(f"Saved scores to {scores_path}")
    print(df[["section_name", "period", "happiness", "labmt_token_count"]].head())

    # 5. 生成新的 summary table（按 section + period 聚合）
    #    这一步就是你想要新增的 table。
    # 5. Generate a new summary table (aggregated by section + period)
    #   This step is the table you want to add.
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
            # NEW: Average coverage per group
            mean_coverage=("labmt_proportion", "mean"),
        )
        .reset_index()
    )

    # 6. 把 summary table 保存到 tables/ 目录
    # 6. Save the summary table to the tables/ directory.
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / "guardian_section_happiness_summary.csv"
    summary.to_csv(table_path, index=False, encoding="utf-8")

    print(f"Saved summary table to {table_path}")
    print(summary.head(8))


if __name__ == "__main__":
    main()