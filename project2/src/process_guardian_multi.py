import json
from pathlib import Path

import pandas as pd

from process_guardian import results_to_dataframe


def main():
    # This should match the NUM_PAGES you used in fetch_guardian_multi.py
    NUM_PAGES = 3

    raw_path = Path(f"data/raw/guardian_results_pages1_{NUM_PAGES}.json")

    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. "
            f"Please run 'python src/fetch_guardian_multi.py' first."
        )

    # 1. Load combined results JSON
    with raw_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        results = data["results"]
    except KeyError:
        raise KeyError("Combined JSON does not contain a top-level 'results' list.")

    print(f"Total number of articles in combined results: {len(results)}")

    if not results:
        print("No articles found in combined results.")
        return

    # 2. Convert to DataFrame (reuse the function from process_guardian.py)
    df = results_to_dataframe(results)
    print("\nDataFrame shape:", df.shape)
    print("DataFrame columns:", list(df.columns))

    # 3. Save to data/processed/
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / f"guardian_articles_pages1_{NUM_PAGES}.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved multi-page processed CSV to: {output_path.resolve()}")


if __name__ == "__main__":
    main()