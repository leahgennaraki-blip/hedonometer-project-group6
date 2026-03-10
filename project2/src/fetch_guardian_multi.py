from pathlib import Path
import json

from fetch_guardian import fetch_guardian_page, save_json


def main():
    # ----- CONFIGURATION -----
    # You can change these numbers to control how many articles you collect.
    NUM_PAGES = 3        # how many pages to fetch
    PAGE_SIZE = 50       # how many articles per page (max 200 in Guardian API)
    QUERY = "politics"
    FROM_DATE = "2020-01-01"
    TO_DATE = "2023-12-31"
    # -------------------------

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for page in range(1, NUM_PAGES + 1):
        print(f"\n=== Fetching page {page} ===")
        data = fetch_guardian_page(
            page=page,
            page_size=PAGE_SIZE,
            query=QUERY,
            from_date=FROM_DATE,
            to_date=TO_DATE,
        )

        # Save full raw JSON for this page
        page_path = raw_dir / f"guardian_page{page}.json"
        save_json(data, page_path)

        # Collect results list for later combination
        results = data.get("response", {}).get("results", [])
        print(f"Page {page}: {len(results)} articles")
        all_results.extend(results)

    # Save a combined results JSON for convenience
    combined_path = raw_dir / f"guardian_results_pages1_{NUM_PAGES}.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, ensure_ascii=False, indent=2)

    print(
        f"\nTotal articles across {NUM_PAGES} pages: {len(all_results)}"
    )
    print(f"Saved combined results JSON to: {combined_path.resolve()}")


if __name__ == "__main__":
    main()