"""データ生成とスキーマ検証のテスト。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from demand_forecast.config import DataConfig
from demand_forecast.data.generate import generate_demand_data
from demand_forecast.data.loaders import (
    DataValidationError,
    read_demand_frame,
    validate_demand_frame,
    write_frame,
)

SMALL_CONFIG = DataConfig(
    start_date=dt.date(2025, 1, 1),
    end_date=dt.date(2025, 3, 31),
    n_stores=2,
    n_skus=3,
)


def test_generated_data_has_expected_shape() -> None:
    df = generate_demand_data(SMALL_CONFIG, seed=1)
    n_days = (SMALL_CONFIG.end_date - SMALL_CONFIG.start_date).days + 1
    assert df.height == n_days * SMALL_CONFIG.n_stores * SMALL_CONFIG.n_skus
    assert df.select(["store_id", "sku_id"]).unique().height == 6


def test_generation_is_reproducible() -> None:
    """同じシードなら完全に同じデータになること。"""
    first = generate_demand_data(SMALL_CONFIG, seed=123)
    second = generate_demand_data(SMALL_CONFIG, seed=123)
    assert first.equals(second)


def test_different_seeds_give_different_data() -> None:
    first = generate_demand_data(SMALL_CONFIG, seed=1)
    second = generate_demand_data(SMALL_CONFIG, seed=2)
    assert not first.equals(second)


def test_adding_a_sku_does_not_change_existing_series() -> None:
    """系列を増やしても、既存系列の値が変わらないこと。

    系列ごとに独立したシードを与えている設計が守られているかの確認。
    ここが崩れると、データ規模を変えるたびに過去の実験結果と比較できなくなる。
    """
    small = generate_demand_data(SMALL_CONFIG, seed=5)
    larger = generate_demand_data(
        DataConfig(
            start_date=SMALL_CONFIG.start_date,
            end_date=SMALL_CONFIG.end_date,
            n_stores=SMALL_CONFIG.n_stores,
            n_skus=SMALL_CONFIG.n_skus + 2,
        ),
        seed=5,
    )
    existing_skus = small.get_column("sku_id").unique().to_list()
    subset = larger.filter(pl.col("sku_id").is_in(existing_skus)).sort(
        ["date", "store_id", "sku_id"]
    )
    assert small.sort(["date", "store_id", "sku_id"]).equals(subset)


def test_generated_data_passes_validation() -> None:
    assert validate_demand_frame(generate_demand_data(SMALL_CONFIG, seed=1)).height > 0


def test_generated_demand_is_non_negative_and_variable() -> None:
    df = generate_demand_data(SMALL_CONFIG, seed=1)
    units = df.get_column("units_sold")
    assert units.min() >= 0
    assert units.std() > 0  # 定数列になっていないこと


def test_promotion_lowers_price() -> None:
    """販促フラグが立っている日は通常より安いこと（生成ロジックの意図の確認）。"""
    df = generate_demand_data(SMALL_CONFIG, seed=1)
    promo_price = df.filter(pl.col("promo_flag") == 1).get_column("price").mean()
    normal_price = df.filter(pl.col("promo_flag") == 0).get_column("price").mean()
    assert promo_price < normal_price


def test_missing_column_is_rejected() -> None:
    df = generate_demand_data(SMALL_CONFIG, seed=1).drop("price")
    with pytest.raises(DataValidationError, match="必須カラムが不足"):
        validate_demand_frame(df)


def test_wrong_dtype_is_rejected() -> None:
    df = generate_demand_data(SMALL_CONFIG, seed=1).with_columns(
        pl.col("units_sold").cast(pl.Float64)
    )
    with pytest.raises(DataValidationError, match="型が想定と異なります"):
        validate_demand_frame(df)


def test_duplicate_keys_are_rejected() -> None:
    df = generate_demand_data(SMALL_CONFIG, seed=1)
    with pytest.raises(DataValidationError, match="重複"):
        validate_demand_frame(pl.concat([df, df.head(1)]))


def test_date_gap_is_rejected() -> None:
    """日付に穴があるデータを弾くこと（ラグが静かにずれるのを防ぐ）。"""
    df = generate_demand_data(SMALL_CONFIG, seed=1)
    with_gap = df.filter(pl.col("date") != dt.date(2025, 2, 10))
    with pytest.raises(DataValidationError, match="連続していない"):
        validate_demand_frame(with_gap)


def test_date_gap_is_allowed_when_not_required() -> None:
    df = generate_demand_data(SMALL_CONFIG, seed=1)
    with_gap = df.filter(pl.col("date") != dt.date(2025, 2, 10))
    assert validate_demand_frame(with_gap, require_contiguous=False).height == with_gap.height


def test_negative_units_are_rejected() -> None:
    df = generate_demand_data(SMALL_CONFIG, seed=1).with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(-1)
        .otherwise(pl.col("units_sold"))
        .cast(pl.Int32)
        .alias("units_sold")
    )
    with pytest.raises(DataValidationError, match="負の値"):
        validate_demand_frame(df)


def test_empty_frame_is_rejected() -> None:
    df = generate_demand_data(SMALL_CONFIG, seed=1).head(0)
    with pytest.raises(DataValidationError, match="空です"):
        validate_demand_frame(df)


def test_roundtrip_through_parquet(tmp_path: Path) -> None:
    df = generate_demand_data(SMALL_CONFIG, seed=1)
    path = write_frame(df, tmp_path / "demand.parquet")
    assert read_demand_frame(path).equals(df.sort(["date", "store_id", "sku_id"]))


def test_missing_file_gives_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="generate-data"):
        read_demand_frame(tmp_path / "missing.parquet")
