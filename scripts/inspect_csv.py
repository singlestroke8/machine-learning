"""提供された CSV を精査するための使い捨てスクリプト。

取り込みコードを書く前に、まず中身を確認するためのもの。
スキーマを一切仮定せず、何が入っているかだけを報告する。

    uv run python scripts/inspect_csv.py path/to/data.csv

想定と違う形（区切り文字、文字コード、ヘッダ位置）でも落ちないよう、
読み込みは緩く行い、判断は人間に任せる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

MAX_UNIQUE_TO_LIST = 15
SAMPLE_ROWS = 5


def _read(path: Path) -> pl.DataFrame:
    """区切り文字と文字コードを推測しながら読み込む。"""
    head = path.read_bytes()[:8192]
    for encoding in ("utf8", "utf8-lossy"):
        for separator in (",", "\t", ";"):
            try:
                df = pl.read_csv(
                    path,
                    separator=separator,
                    encoding=encoding,
                    infer_schema_length=10000,
                    try_parse_dates=True,
                )
            except Exception:
                continue
            if df.width > 1:
                print(f"読み込み: separator={separator!r}, encoding={encoding!r}")
                return df
    msg = f"CSV として読めませんでした。先頭バイト: {head[:200]!r}"
    raise SystemExit(msg)


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("使い方: uv run python scripts/inspect_csv.py <csv path>")

    path = Path(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"ファイルが見つかりません: {path}")

    print(f"ファイル: {path} ({path.stat().st_size / 1_048_576:.2f} MB)")
    df = _read(path)

    _section("1. 形")
    print(f"行数: {df.height:,} / 列数: {df.width}")

    _section("2. 列ごとの型・欠損・ユニーク数")
    rows = []
    for name in df.columns:
        col = df.get_column(name)
        n_null = int(col.null_count())
        rows.append(
            {
                "column": name,
                "dtype": str(col.dtype),
                "null": n_null,
                "null_pct": round(100 * n_null / df.height, 2) if df.height else 0.0,
                "n_unique": int(col.n_unique()),
            }
        )
    with pl.Config(tbl_rows=-1, fmt_str_lengths=40, tbl_cols=-1):
        print(pl.DataFrame(rows))

    _section(f"3. 先頭 {SAMPLE_ROWS} 行")
    with pl.Config(tbl_rows=SAMPLE_ROWS, tbl_cols=-1, fmt_str_lengths=30):
        print(df.head(SAMPLE_ROWS))

    _section("4. 数値列の要約")
    numeric = [c for c in df.columns if df.schema[c].is_numeric()]
    if numeric:
        with pl.Config(tbl_rows=-1, tbl_cols=-1):
            print(df.select(numeric).describe())
    else:
        print("（数値列なし）")

    _section("5. カテゴリらしい列の値")
    for name in df.columns:
        col = df.get_column(name)
        n_unique = int(col.n_unique())
        if 0 < n_unique <= MAX_UNIQUE_TO_LIST:
            values = col.unique().sort().to_list()
            print(f"{name} ({n_unique}種): {values}")

    _section("6. 日付らしい列")
    date_like = [
        c
        for c in df.columns
        if df.schema[c] in (pl.Date, pl.Datetime)
        or any(k in c.lower() for k in ("date", "day", "time", "ymd", "日付", "年月"))
    ]
    if not date_like:
        print("（日付列を自動検出できず。列名から手で指定が必要）")
    for name in date_like:
        col = df.get_column(name)
        print(f"\n--- {name} (dtype={col.dtype}) ---")
        print(f"  最小: {col.min()}  最大: {col.max()}")
        print(f"  ユニーク日数: {col.n_unique():,}")
        if df.schema[name] in (pl.Date, pl.Datetime):
            span = (
                df.select(
                    (pl.col(name).max() - pl.col(name).min()).dt.total_days().alias("d")
                ).item()
                or 0
            )
            print(f"  期間の日数: {span + 1:,}（実データの日数との差 = 欠落日数）")

    _section("7. 重複の可能性")
    print(f"完全重複行: {df.height - df.unique().height:,}")

    _section("次の判断ポイント")
    print("""
以下を確認してから取り込みコードを書きます。

  a) 系列を識別する列はどれか（店舗ID・商品IDに相当するもの）
  b) 予測対象（実績数量）はどれか
  c) 日付列はどれか。粒度は日次か、週次・月次か
  d) 日付の欠落があるか（上の「期間の日数」と「ユニーク日数」の差）
  e) 価格・販促に相当する列があるか（無ければ特徴量設計を変える）
  f) 欠品・返品・マイナス値の扱い
""")


if __name__ == "__main__":
    main()
