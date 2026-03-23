from pathlib import Path
import re
from collections import Counter

import pandas as pd

from load_labmt import load_labmt

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z']+")
NEUTRAL_MIN = 4.0
NEUTRAL_MAX = 6.0
TOP_N = 15
MIN_LENGTH = 6  # <-- only words longer than 6 letters

TARGET_SECTIONS = ["Politics", "World news", "Opinion"]


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


def keep_sentiment_words(word_to_score: dict) -> dict:
    """
    Keep only sentiment-bearing words:
    score < 4 OR score > 6
    """
    return {
        word: score
        for word, score in word_to_score.items()
        if score < NEUTRAL_MIN or score > NEUTRAL_MAX
    }


def top_words_with_scores(
    df_subset: pd.DataFrame,
    filtered_lexicon: dict,
    n: int = 15
) -> pd.DataFrame:
    """
    Count most frequent sentiment words with length filter
    """
    counter = Counter()

    for text in df_subset["body_text"]:
        tokens = tokenize(text)
        counter.update(
            w for w in tokens
            if w in filtered_lexicon and len(w) > MIN_LENGTH
        )

    rows = []
    for word, count in counter.most_common(n):
        rows.append({
            "word": word,
            "count": count,
            "score": filtered_lexicon[word]
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
    df = df.dropna(subset=["happiness", "section_name", "body_text"]).copy()
    df = df[df["section_name"].isin(TARGET_SECTIONS)].copy()

    print("\nArticles per section:")
    print(df.groupby("section_name").size())

    # Load lexicon
    labmt_df = load_labmt(labmt_path)
    full_lexicon = dict(zip(labmt_df["word"], labmt_df["happiness_average"]))
    filtered_lexicon = keep_sentiment_words(full_lexicon)

    print(f"\nFiltered lexicon size: {len(filtered_lexicon)}")

    # Get top words per section
    section_tables = {}
    for section in TARGET_SECTIONS:
        df_section = df[df["section_name"] == section]
        section_tables[section] = top_words_with_scores(
            df_section,
            filtered_lexicon,
            n=TOP_N
        ).reset_index(drop=True)

    # Build comparison table
    rows = []
    for i in range(TOP_N):
        row = {"rank": i + 1}

        # Politics
        pol = section_tables["Politics"]
        row["politics_word"] = pol.loc[i, "word"] if i < len(pol) else None

        # World News
        world = section_tables["World news"]
        row["world_news_word"] = world.loc[i, "word"] if i < len(world) else None

        # Opinion
        op = section_tables["Opinion"]
        row["opinion_word"] = op.loc[i, "word"] if i < len(op) else None

        rows.append(row)

    final_df = pd.DataFrame(rows)

    # Save CSV
    out_dir = Path("tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "guardian_section_filtered_long_words.csv"
    final_df.to_csv(out_path, index=False)

    print(f"\nSaved to: {out_path}")

    # Clean table for README
    clean_df = final_df[[
        "politics_word",
        "world_news_word",
        "opinion_word"
    ]]

    print("\nMarkdown version for README:\n")
    print(clean_df.to_markdown(index=False))


if __name__ == "__main__":
    main()