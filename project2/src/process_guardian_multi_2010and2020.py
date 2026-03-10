# src/process_guardian.py
import json
from pathlib import Path

import pandas as pd


def load_results_from_dir(raw_dir: Path, period_label: str):
    """
    从某个 raw 目录里把所有 *_page*.json 读出来，拉平成 list[dict]。
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


def main():
    raw_2010 = Path("data/raw/guardian_2010_2013")
    raw_2020 = Path("data/raw/guardian_2020_2023")

    records_2010 = load_results_from_dir(raw_2010, "2010-2013")
    records_2020 = load_results_from_dir(raw_2020, "2020-2023")

    all_records = records_2010 + records_2020
    df = pd.DataFrame(all_records)

    # 去重（有时多页会有重复 id）
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_path = processed_dir / "guardian_articles_processed.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved processed data to: {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()