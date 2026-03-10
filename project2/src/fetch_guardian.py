import os
import json
from pathlib import Path

import requests

# 我们约定统一使用这个环境变量名来存 Guardian 的 key
API_KEY_ENV_VAR = "GUARDIAN_API_KEY"


def get_api_key() -> str:
    """
    从环境变量读取 Guardian API key。
    如果没有设置，则抛出一个带有清晰提示的错误。
    """
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"没有找到环境变量 {API_KEY_ENV_VAR}。\n"
            f"请在终端中执行：\n"
            f'export {API_KEY_ENV_VAR}="你的key"\n'
            f"然后再运行：python src/fetch_guardian.py"
        )
    return api_key


def fetch_guardian_page(
    page: int = 1,
    page_size: int = 100,
    query: str = "politics",
    from_date: str = "2023-01-01",
    to_date: str = "2023-12-31",
) -> dict:
    """
    调用 Guardian Content API，获取一页搜索结果并返回 JSON 数据。
    """
    api_key = get_api_key()
    base_url = "https://content.guardianapis.com/search"

    params = {
        "api-key": api_key,
        "q": query,
        "from-date": from_date,
        "to-date": to_date,
        "page": page,
        "page-size": page_size,
        "show-fields": "headline,trailText,bodyText",
    }

    print(f"Requesting Guardian page {page} with query='{query}' ...")

    try:
        response = requests.get(base_url, params=params, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"请求 Guardian API 失败（网络错误）：{e}")

    # 如果 HTTP 状态码不是 200，抛出详细错误
    if response.status_code != 200:
        raise RuntimeError(
            f"Guardian API 返回非 200 状态码：{response.status_code}\n"
            f"响应内容：\n{response.text}"
        )

    data = response.json()

    # 进一步检查 API 自己的状态字段
    api_status = data.get("response", {}).get("status")
    if api_status != "ok":
        raise RuntimeError(
            f"Guardian API 响应 status != 'ok'：{api_status}\n"
            f"完整响应：\n{json.dumps(data, ensure_ascii=False, indent=2)}"
        )

    return data


def save_json(data: dict, path: Path) -> None:
    """
    将 JSON 数据保存到指定路径（自动创建父目录）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved Guardian API response to {path.resolve()}")


def inspect_results(data: dict) -> None:
    """
    打印一些简单信息，帮助我们确认抓到的数据内容。
    """
    try:
        results = data["response"]["results"]
    except KeyError:
        print("响应中没有找到 ['response']['results'] 字段。")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    print(f"Number of articles in this page: {len(results)}")
    if results:
        first = results[0]
        print("Example article id:", first.get("id"))
        print("Example webTitle:", first.get("webTitle"))


def main():
    output_path = Path("data/raw/guardian_sample_page1.json")

    data = fetch_guardian_page(page=1, page_size=100, query="politics")
    save_json(data, output_path)
    inspect_results(data)


if __name__ == "__main__":
    main()