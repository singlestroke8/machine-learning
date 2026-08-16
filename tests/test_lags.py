"""origin 特徴量のテスト。

ラグの意味がずれていても例外は出ないため、手で計算できる小さなデータで
「何日前の値が入っているか」を1つずつ確認する。
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from demand_forecast.features.lags import (
    add_origin_features,
    all_same_dow_columns,
    same_dow_columns,
    same_dow_offset,
)


@pytest.fixture
def toy_frame() -> pl.DataFrame:
    """1系列・30日ぶんの、値が日付と対応づく分かりやすいデータ。"""
    n = 30
    start = dt.date(2026, 1, 1)
    return pl.DataFrame(
        {
            "date": [start + dt.timedelta(days=i) for i in range(n)],
            "store_id": ["S01"] * n,
            "sku_id": ["SKU01"] * n,
            "units_sold": list(range(1, n + 1)),
            "price": [100.0] * n,
            "promo_flag": [0] * n,
        },
        schema={
            "date": pl.Date,
            "store_id": pl.Utf8,
            "sku_id": pl.Utf8,
            "units_sold": pl.Int32,
            "price": pl.Float64,
            "promo_flag": pl.Int8,
        },
    )


def test_lag_1_is_the_origin_day_itself(toy_frame: pl.DataFrame) -> None:
    """org_lag_1 は origin 当日の実績であること。"""
    out = add_origin_features(toy_frame, lags=[1, 2, 7], windows=[7])
    row = out.filter(pl.col("date") == dt.date(2026, 1, 20))
    assert row.get_column("org_lag_1").item() == 20.0
    assert row.get_column("org_lag_2").item() == 19.0
    assert row.get_column("org_lag_7").item() == 14.0


def test_rolling_mean_window_includes_origin(toy_frame: pl.DataFrame) -> None:
    """移動平均の窓が origin 当日を含む直近 w 日であること。"""
    out = add_origin_features(toy_frame, lags=[1], windows=[7])
    row = out.filter(pl.col("date") == dt.date(2026, 1, 20))
    # 14..20 の平均
    assert row.get_column("org_roll_mean_7").item() == pytest.approx(17.0)


def test_early_rows_are_null_not_silently_filled(toy_frame: pl.DataFrame) -> None:
    """履歴が足りない先頭行は、勝手に埋めずに欠損のままにすること。"""
    out = add_origin_features(toy_frame, lags=[1], windows=[7])
    first = out.filter(pl.col("date") == dt.date(2026, 1, 3))
    assert first.get_column("org_roll_mean_7").item() is None


def test_features_do_not_leak_across_series() -> None:
    """系列をまたいで値が混ざらないこと。"""
    n = 10
    start = dt.date(2026, 1, 1)
    frame = pl.concat(
        [
            pl.DataFrame(
                {
                    "date": [start + dt.timedelta(days=i) for i in range(n)],
                    "store_id": ["S01"] * n,
                    "sku_id": [sku] * n,
                    "units_sold": [value] * n,
                    "price": [100.0] * n,
                    "promo_flag": [0] * n,
                },
                schema={
                    "date": pl.Date,
                    "store_id": pl.Utf8,
                    "sku_id": pl.Utf8,
                    "units_sold": pl.Int32,
                    "price": pl.Float64,
                    "promo_flag": pl.Int8,
                },
            )
            for sku, value in (("SKU01", 5), ("SKU02", 500))
        ]
    )
    out = add_origin_features(frame, lags=[1, 2], windows=[3])
    for sku, value in (("SKU01", 5.0), ("SKU02", 500.0)):
        row = out.filter((pl.col("sku_id") == sku) & (pl.col("date") == dt.date(2026, 1, 10)))
        assert row.get_column("org_roll_mean_3").item() == pytest.approx(value)


@pytest.mark.parametrize(
    ("horizon", "expected"),
    [(7, 0), (14, 0), (1, 6), (2, 5), (6, 1), (8, 6)],
)
def test_same_dow_offset(horizon: int, expected: int) -> None:
    """target と同じ曜日が origin の何日前かの計算が合っていること。"""
    assert same_dow_offset(horizon) == expected


def test_same_dow_offset_matches_actual_weekday(toy_frame: pl.DataFrame) -> None:
    """計算した offset の日が、本当に target と同じ曜日になっていること。"""
    origin = dt.date(2026, 1, 20)
    for horizon in range(1, 15):
        target = origin + dt.timedelta(days=horizon)
        reference = origin - dt.timedelta(days=same_dow_offset(horizon))
        assert reference.weekday() == target.weekday()


def test_same_dow_columns_are_subset_of_all(toy_frame: pl.DataFrame) -> None:
    """horizon ごとに選ぶカラムが、実際に生成されるカラムに含まれること。"""
    out = add_origin_features(toy_frame, lags=[1], windows=[7])
    generated = set(out.columns)
    assert set(all_same_dow_columns()) <= generated
    for horizon in range(1, 15):
        assert set(same_dow_columns(horizon)) <= generated
