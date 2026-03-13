# src/fetch_guardian_multi_2020.py
from pathlib import Path

from fetch_guardian_common import fetch_many_pages


def main():
    # ---- 配置区域 ----
    NUM_PAGES = 3
    PAGE_SIZE = 50
    QUERY = "politics"        # 与上面保持一致，或根据研究问题调整
    FROM_DATE = "2020-01-01"
    TO_DATE = "2023-12-31"
    LABEL = "2020_2023"

    raw_dir = Path("data/raw/guardian_2020_2023")

    fetch_many_pages(
        out_dir=raw_dir,
        from_date=FROM_DATE,
        to_date=TO_DATE,
        query=QUERY,
        num_pages=NUM_PAGES,
        page_size=PAGE_SIZE,
        label=LABEL,
    )


if __name__ == "__main__":
    main()