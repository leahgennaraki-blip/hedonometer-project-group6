from pathlib import Path
import re
from collections import Counter

import pandas as pd

from load_labmt import load_labmt

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z']+")
NEGATIVE_THRESHOLD = 4.0
TOP_N = 10
TARGET_PERIODS = ["2010-2013", "2020-2023"]


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


def keep_negative_words(word_to_score: dict) -> dict:
    """
    Keep only negative labMT words:
    score < 4.0
    """
    return {
        word: score
        for word, score in word_to_score.items()
        if score < NEGATIVE_THRESHOLD
    }


def top_words_with_scores(df_subset: pd.DataFrame, negative_lexicon: dict, n: int = 10) -> pd.DataFrame:
    """
    Count the most frequent remaining negative lexicon words in a subset of articles.
    Returns: word, count, score
    """
    counter = Counter()

    for text in df_subset["body_text"]:
        tokens = tokenize(text)
        counter.update(w for w in tokens if w in negative_lexicon)

    rows = []
    for word, count in counter.most_common(n):
        rows.append({
            "word": word,
            "count": count,
            "score": negative_lexicon[word]
        })

    return pd.DataFrame(rows)


def resolve_existing_path(candidates):
    for path_str in candidates:
        p = Path(path_str)
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    articles_path = resolve_existing_path([
        "data/processed/guardian_articles_with_scores.csv",
        "project2/data/processed/guardian_articles_with_scores.csv",
    ])

    labmt_path = resolve_existing_path([
        "data/raw/Data_Set_S1.txt",
        "project2/data/raw/Data_Set_S1.txt",
    ])

    print(f"Using articles file: {articles_path}")
    print(f"Using labMT file: {labmt_path}")

    # Load articles
    df = pd.read_csv(articles_path)
    df = df.dropna(subset=["happiness"]).copy()

    print("\nArticles per period:")
    print(df.groupby("period").size())

    # Load labMT
    labmt_df = load_labmt(labmt_path)
    full_lexicon = dict(zip(labmt_df["word"], labmt_df["happiness_average"]))
    negative_lexicon = keep_negative_words(full_lexicon)

    print(f"\nFull labMT lexicon size: {len(full_lexicon)}")
    print(f"Negative-only lexicon size: {len(negative_lexicon)}")

    # Top 10 negative words per overall period
    period_tables = {}
    for period in TARGET_PERIODS:
        df_period = df[df["period"] == period]
        period_tables[period] = top_words_with_scores(df_period, negative_lexicon, n=TOP_N).reset_index(drop=True)

    # Side-by-side comparison table
    rows = []
    for i in range(TOP_N):
        row = {"rank": i + 1}

        a = period_tables["2010-2013"]
        if i < len(a):
            row["2010_2013_word"] = a.loc[i, "word"]
            row["2010_2013_count"] = a.loc[i, "count"]
            row["2010_2013_score"] = a.loc[i, "score"]
        else:
            row["2010_2013_word"] = None
            row["2010_2013_count"] = None
            row["2010_2013_score"] = None

        b = period_tables["2020-2023"]
        if i < len(b):
            row["2020_2023_word"] = b.loc[i, "word"]
            row["2020_2023_count"] = b.loc[i, "count"]
            row["2020_2023_score"] = b.loc[i, "score"]
        else:
            row["2020_2023_word"] = None
            row["2020_2023_count"] = None
            row["2020_2023_score"] = None

        rows.append(row)

    final_df = pd.DataFrame(rows)

    # Save CSV
    out_dir = Path("tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "guardian_overall_negative_word_exhibit.csv"
    final_df.to_csv(out_path, index=False)

    print(f"\nSaved comparison table to: {out_path}")
    print("\nPreview:")
    print(final_df.to_string(index=False))

    print("\nMarkdown version for README:\n")
    print(final_df.to_markdown(index=False))


if __name__ == "__main__":
    main()