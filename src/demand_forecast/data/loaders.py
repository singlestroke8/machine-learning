"""データの入出力とスキーマ検証。

「読み込んだ直後に落とす」ことを重視している。時系列の特徴量生成は
日付が1日刻みで欠落していないことを暗黙の前提にしており、そこが崩れると
ラグが静かにずれて、精度だけが説明不能に悪化する。壊れたデータで学習を
完走させるより、その場で例外にしたほうが調査コストが小さい。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

DEMAND_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date(),
    "store_id": pl.Utf8(),
    "sku_id": pl.Utf8(),
    "units_sold": pl.Int32(),
    "price": pl.Float64(),
    "promo_flag": pl.Int8(),
}

_KEY_COLS = ["date", "store_id", "sku_id"]


class DataValidationError(ValueError):
    """入力データがスキーマ・整合性の要件を満たさない場合に送出する。"""


def validate_demand_frame(df: pl.DataFrame, *, require_contiguous: bool = True) -> pl.DataFrame:
    """需要データの構造と整合性を検証する。

    Args:
        df: 検証対象。
        require_contiguous: 系列ごとに日付が1日刻みで連続していることを
            要求するか。推論時の履歴など、部分的なデータには ``False`` を渡す。

    Returns:
        検証を通過した DataFrame（キー順にソート済み）。

    Raises:
        DataValidationError: 要件を満たさない場合。
    """
    missing = [c for c in DEMAND_SCHEMA if c not in df.columns]
    if missing:
        msg = f"必須カラムが不足しています: {missing}（存在するカラム: {df.columns}）"
        raise DataValidationError(msg)

    if df.is_empty():
        msg = "データが空です。"
        raise DataValidationError(msg)

    for col, dtype in DEMAND_SCHEMA.items():
        if df.schema[col] != dtype:
            msg = f"カラム {col!r} の型が想定と異なります: 期待={dtype}, 実際={df.schema[col]}"
            raise DataValidationError(msg)

    null_counts = {
        col: int(df.get_column(col).null_count())
        for col in DEMAND_SCHEMA
        if df.get_column(col).null_count() > 0
    }
    if null_counts:
        msg = f"欠損値が含まれています: {null_counts}"
        raise DataValidationError(msg)

    n_duplicated = df.select(_KEY_COLS).is_duplicated().sum()
    if n_duplicated:
        msg = f"(date, store_id, sku_id) に重複が {n_duplicated} 行あります。"
        raise DataValidationError(msg)

    if df.get_column("units_sold").min() < 0:  # type: ignore[operator]
        msg = "units_sold に負の値が含まれています。"
        raise DataValidationError(msg)

    if df.get_column("price").min() <= 0:  # type: ignore[operator]
        msg = "price に 0 以下の値が含まれています。"
        raise DataValidationError(msg)

    invalid_promo = df.filter(~pl.col("promo_flag").is_in([0, 1])).height
    if invalid_promo:
        msg = f"promo_flag が 0/1 以外の行が {invalid_promo} 行あります。"
        raise DataValidationError(msg)

    if require_contiguous:
        _assert_contiguous_dates(df)

    return df.sort(_KEY_COLS)


def _assert_contiguous_dates(df: pl.DataFrame) -> None:
    """系列ごとに日付が1日刻みで連続していることを確認する。"""
    gaps = (
        df.sort(["store_id", "sku_id", "date"])
        .with_columns(
            (pl.col("date") - pl.col("date").shift(1).over(["store_id", "sku_id"]))
            .dt.total_days()
            .alias("_gap")
        )
        .filter(pl.col("_gap").is_not_null() & (pl.col("_gap") != 1))
    )
    if not gaps.is_empty():
        sample = gaps.select(["store_id", "sku_id", "date", "_gap"]).head(5).to_dicts()
        msg = f"日付が連続していない系列があります（{gaps.height} 箇所）。 先頭のみ表示: {sample}"
        raise DataValidationError(msg)


def read_demand_frame(path: Path | str, *, validate: bool = True) -> pl.DataFrame:
    """Parquet から需要データを読み込む。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
    """
    file_path = Path(path)
    if not file_path.exists():
        msg = (
            f"データファイルが見つかりません: {file_path}\n"
            "先に `uv run dfc generate-data` を実行してください。"
        )
        raise FileNotFoundError(msg)
    df = pl.read_parquet(file_path)
    return validate_demand_frame(df) if validate else df


def write_frame(df: pl.DataFrame, path: Path | str) -> Path:
    """DataFrame を Parquet として保存する（親ディレクトリは自動作成）。"""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(file_path)
    return file_path
