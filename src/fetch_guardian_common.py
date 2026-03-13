# src/fetch_guardian.py
import os
import time
import json
from pathlib import Path

import requests

BASE_URL = "https://content.guardianapis.com/search"
API_KEY = os.getenv("GUARDIAN_API_KEY")


def fetch_guardian_page(page, page_size, from_date, to_date, query):
    """
    调 Guardian Content API 抓取一页结果。
    """
    if API_KEY is None:
        raise RuntimeError(
            "环境变量 GUARDIAN_API_KEY 没有设置，请在终端里先 export GUARDIAN_API_KEY=你的key"
        )

    params = {
        "api-key": API_KEY,
        "from-date": from_date,          # 例如 "2010-01-01"
        "to-date": to_date,              # 例如 "2013-12-31"
        "page": page,
        "page-size": page_size,          # 这里用 50
        "q": query,                      # 例如 "politics"
        "show-fields": "headline,trailText,bodyText",
        "show-tags": "keyword,section",
        "order-by": "relevance",         # 或 "newest"/"oldest"，按你们研究问题改
    }

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # 简单检查
    if data.get("response", {}).get("status") != "ok":
        raise RuntimeError(f"Guardian API 返回非 ok 状态: {data}")

    return data


def save_json(obj, path: Path):
    """
    以 UTF-8 + 缩进格式保存 JSON，方便以后调试。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_many_pages(
    out_dir: Path,
    from_date: str,
    to_date: str,
    query: str,
    num_pages: int = 3,
    page_size: int = 50,
    label: str | None = None,
):
    """
    连续抓 num_pages 页并保存到 out_dir，同时返回所有 results 的列表。
    """
    if label is None:
        label = f"{from_date}_{to_date}"

    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for page in range(1, num_pages + 1):
        print(f"[{label}] Fetching page {page}/{num_pages} ...")
        data = fetch_guardian_page(
            page=page,
            page_size=page_size,
            from_date=from_date,
            to_date=to_date,
            query=query,
        )

        # 每一页单独保存
        page_path = out_dir / f"guardian_{label}_page{page}.json"
        save_json(data, page_path)

        # 收集 results
        results = data.get("response", {}).get("results", [])
        all_results.extend(results)

        # 简单限流，避免 API 限制
        time.sleep(0.3)

    # 也保存一个汇总文件，方便后面处理
    all_path = out_dir / f"guardian_{label}_all_results.json"
    save_json({"results": all_results}, all_path)

    print(f"[{label}] Done. Total articles collected: {len(all_results)}")
    return all_results