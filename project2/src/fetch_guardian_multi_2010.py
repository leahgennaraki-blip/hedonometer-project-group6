# src/fetch_guardian_multi_2010.py
from pathlib import Path

from fetch_guardian import fetch_many_pages


def main():
    # ---- 配置区域（可根据研究问题修改） ----
    NUM_PAGES = 3          # 抓 3 页
    PAGE_SIZE = 50         # 每页 50 篇
    QUERY = "politics"     # 你们研究的关键词，可以改成别的
    FROM_DATE = "2010-01-01"
    TO_DATE = "2013-12-31"
    LABEL = "2010_2013"

    raw_dir = Path("data/raw/guardian_2010_2013")

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