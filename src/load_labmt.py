from __future__ import annotations

from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# 路径设置：相对当前脚本自动找到 Data_Set_S1.txt
# -----------------------------------------------------------------------------

# 当前文件位置：.../hedonometer-project-group6/src/load_labmt.py
# parents[0] -> src
# parents[1] -> hedonometer-project-group6（项目根目录）
BASE_DIR = Path(__file__).resolve().parents[1]

# 原始数据文件：data/raw/Data_Set_S1.txt
RAW_PATH = BASE_DIR / "data" / "raw" / "Data_Set_S1.txt"


def load_labmt(path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Load labMT 1.0 tab-delimited file.

    Assumptions from assignment:
    - File may contain metadata/comment lines before the header.
    - Missing ranks are represented as '--'.
    """

    # 1. 先用纯文本读取整个文件，用来“自动找到表头在哪一行”
    if not path.exists():
        raise FileNotFoundError(f"Cannot find data file at: {path}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # 2. 启发式找表头：包含 'word' 和 'happiness' 的那一行
    header_idx: int | None = None
    for i, line in enumerate(lines[:200]):  # 表头一般在前 200 行之内
        if "\t" in line and "word" in line.lower() and "happiness" in line.lower():
            header_idx = i
            break

    if header_idx is None:
        # 如果找不到，说明文件结构跟预期不一样，需要人工检查
        raise ValueError("Could not find header row. Inspect the raw file and adjust header detection.")

    # 3. 从表头开始，用 pandas 读表
    df = pd.read_csv(
        path,
        sep="\t",              # 制表符分隔
        skiprows=header_idx,   # 跳过上面的 metadata 行
        na_values=["--"],      # 把 '--' 视为缺失值 NaN
        keep_default_na=True,
        dtype=str,             # 先全部读成字符串，后面再转成数值
    )

    # 4. 列名去掉前后空格
    df.columns = [c.strip() for c in df.columns]

    # 5. 除了 'word' 以外的列都尝试转为数值
    numeric_cols = [c for c in df.columns if c != "word"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 6. 'word' 列保证是字符串，并去掉前后空格
    df["word"] = df["word"].astype(str).str.strip()

    return df


def main() -> None:
    """简单跑一遍 1.1–1.3 的检查，方便单独测试这个文件。"""
    df = load_labmt()

    print("Shape (rows, cols):", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nMissing values per column:\n", df.isna().sum().sort_values(ascending=False))
    print("\nDuplicate words:", df["word"].duplicated().sum())
    print("\nSample rows:\n", df.sample(15, random_state=42))


if __name__ == "__main__":
    main()