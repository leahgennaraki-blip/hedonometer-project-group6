import json
from pathlib import Path

import pandas as pd


def load_raw_json() -> dict:
    """Load the raw Guardian JSON file from data/raw/."""
    raw_path = Path("data/raw/guardian_sample_page1.json")

    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Please run "
            f"'python src/fetch_guardian.py' first to download data."
        )

    with raw_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def inspect_first_article(results: list) -> None:
    """Print some information about the first article."""
    if not results:
        print("The results list is empty; no articles found.")
        return

    first = results[0]
    print("\n=== Some information on the first article ===")
    print("id:", first.get("id"))
    print("sectionName:", first.get("sectionName"))
    print("webPublicationDate:", first.get("webPublicationDate"))
    print("webTitle:", first.get("webTitle"))

    fields = first.get("fields", {})
    headline = fields.get("headline", "")
    body_text = fields.get("bodyText", "")

    print("\nheadline:", headline[:100])
    print("\nbodyText (first 200 characters):")
    print(body_text[:200])


def results_to_dataframe(results: list) -> pd.DataFrame:
    """
    Convert the Guardian 'results' list into a tidy pandas DataFrame.
    We keep only the columns that are useful for later analysis.
    """
    records = []

    for item in results:
        fields = item.get("fields", {}) or {}

        record = {
            "id": item.get("id"),
            "type": item.get("type"),
            "section_id": item.get("sectionId"),
            "section_name": item.get("sectionName"),
            "web_publication_date": item.get("webPublicationDate"),
            "web_title": item.get("webTitle"),
            "headline": fields.get("headline"),
            "trail_text": fields.get("trailText"),
            "body_text": fields.get("bodyText"),
            "is_hosted": item.get("isHosted"),
            "pillar_id": item.get("pillarId"),
            "pillar_name": item.get("pillarName"),
        }

        records.append(record)

    df = pd.DataFrame.from_records(records)
    return df


def main():
    # 1. Load raw JSON and extract results list
    data = load_raw_json()

    try:
        results = data["response"]["results"]
    except KeyError:
        raise KeyError("JSON does not contain ['response']['results'].")

    print(f"The total number of articles in this page: {len(results)}")

    # 2. Inspect the first article (for understanding/debugging)
    inspect_first_article(results)

    # 3. Convert to DataFrame
    df = results_to_dataframe(results)
    print("\nDataFrame shape:", df.shape)
    print("DataFrame columns:", list(df.columns))

    # 4. Save to data/processed/
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / "guardian_articles_page1.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved processed CSV to: {output_path.resolve()}")


if __name__ == "__main__":
    main()