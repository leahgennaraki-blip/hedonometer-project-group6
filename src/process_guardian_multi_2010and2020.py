# src/process_guardian.py
import json
from pathlib import Path
from difflib import SequenceMatcher  # <<< ADDED: for text similarity

import pandas as pd


def load_results_from_dir(raw_dir: Path, period_label: str):
    """
    Read all *_page*.json files from a raw directory and flatten them into a list[dict].
    """
    records = []

    for path in raw_dir.glob("guardian_*_page*.json"):
        print(f"Processing {path} ...")
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        for item in data.get("response", {}).get("results", []):
            fields = item.get("fields", {}) or {}

            records.append(
                {
                    "id": item.get("id"),
                    "section_name": item.get("sectionName"),
                    "web_url": item.get("webUrl"),
                    "pub_date": item.get("webPublicationDate"),
                    "headline": fields.get("headline"),
                    "trail_text": fields.get("trailText"),
                    "body_text": fields.get("bodyText"),
                    "period": period_label,
                }
            )

    return records


def calc_text_similarity(a: str, b: str) -> float:  # <<< ADDED: new helper function
    """
    Character-level similarity in [0.0, 1.0]; 1.0 means identical strings.
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def main():
    raw_2010 = Path("data/raw/guardian_2010_2013")
    raw_2020 = Path("data/raw/guardian_2020_2023")

    records_2010 = load_results_from_dir(raw_2010, "2010-2013")
    records_2020 = load_results_from_dir(raw_2020, "2020-2023")

    all_records = records_2010 + records_2020
    df = pd.DataFrame(all_records)

    # Remove exact duplicates by id (sometimes the same item appears on multiple pages)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # <<< ADDED BLOCK: near-duplicate removal based on text similarity
    # Build a combined text field for comparison
    df["full_text"] = (
        df["headline"].fillna("").astype(str)
        + "\n\n"
        + df["body_text"].fillna("").astype(str)
    )

    # Sort so that similar articles are more likely to be adjacent
    df = df.sort_values(["headline", "pub_date"]).reset_index(drop=True)

    similarity_threshold = 0.85  # 0.85 ≈ 85% similarity
    index_to_drop = []

    # Compare only within the same headline to reduce comparisons
    for headline, group in df.groupby("headline"):
        prev_text = None
        prev_idx = None

        for idx, text in group["full_text"].items():
            if prev_text is None:
                prev_text = text
                prev_idx = idx
                continue

            similarity = calc_text_similarity(prev_text, text)

            if similarity >= similarity_threshold:
                # Current row is highly similar to the previous one -> drop current row
                index_to_drop.append(idx)
            else:
                # Update "previous" row
                prev_text = text
                prev_idx = idx

    if index_to_drop:
        df = df.drop(index=index_to_drop).reset_index(drop=True)

    # Drop helper column
    df = df.drop(columns=["full_text"])
    # >>> END OF ADDED BLOCK

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_path = processed_dir / "guardian_articles_processed.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved processed data to: {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()