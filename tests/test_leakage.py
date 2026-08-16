"""リーク検査。このリポジトリで最も重要なテスト。

時系列予測で最も起こりやすく、最も気づきにくい事故が
「予測時点では知り得ない情報を特徴量に混ぜてしまう」ことである。
これが起きると検証スコアだけが良くなり、本番で初めて精度が出ないと分かる。

ここでは実装の中身を読んで確認するのではなく、
**未来の実績を書き換えても、過去の特徴量が1ビットも変わらないこと**
を外から観測して確かめる。実装をどう変えてもこの性質は保たれるべきなので、
リファクタリングに強い検査になる。
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from demand_forecast.config import FeatureConfig
from demand_forecast.features.pipeline import (
    build_inference_frame,
    build_training_frame,
    feature_columns,
)

CUTOFF = dt.date(2025, 1, 31)


def test_future_actuals_do_not_change_past_features(
    demand_frame: pl.DataFrame, feature_config: FeatureConfig
) -> None:
    """cutoff より後の実績を改変しても、origin が cutoff 以前の行は不変であること。"""
    original = build_training_frame(demand_frame, feature_config)

    # cutoff より後の販売数量を 10 倍に改変する
    tampered_demand = demand_frame.with_columns(
        pl.when(pl.col("date") > CUTOFF)
        .then(pl.col("units_sold") * 10)
        .otherwise(pl.col("units_sold"))
        .cast(pl.Int32)
        .alias("units_sold")
    )
    tampered = build_training_frame(tampered_demand, feature_config)

    keys = ["date", "store_id", "sku_id", "feat_horizon"]
    features = [c for c in feature_columns(original) if c not in keys]

    # origin が cutoff 以前 = 予測時点で改変部分をまだ観測していない行
    before = original.filter(pl.col("origin_date") <= CUTOFF).sort(keys).select([*keys, *features])
    after = tampered.filter(pl.col("origin_date") <= CUTOFF).sort(keys).select([*keys, *features])

    assert before.height > 0, "検査対象の行がありません（cutoff の設定を見直すこと）"
    assert before.equals(after), (
        "未来の実績を書き換えたら過去の特徴量が変わりました。"
        " origin より後の情報が特徴量に混入しています。"
    )


def test_features_change_when_pre_origin_actuals_change(
    demand_frame: pl.DataFrame, feature_config: FeatureConfig
) -> None:
    """逆に、origin 以前の実績を変えたら特徴量は変わること。

    上のテストだけでは「特徴量が実績を全く見ていない」場合も通ってしまうため、
    対になる検査を置いて、テスト自体が空振りしていないことを保証する。
    """
    original = build_training_frame(demand_frame, feature_config)
    tampered_demand = demand_frame.with_columns(
        pl.when(pl.col("date") <= CUTOFF)
        .then(pl.col("units_sold") * 10)
        .otherwise(pl.col("units_sold"))
        .cast(pl.Int32)
        .alias("units_sold")
    )
    tampered = build_training_frame(tampered_demand, feature_config)

    keys = ["date", "store_id", "sku_id", "feat_horizon"]
    before = original.filter(pl.col("origin_date") <= CUTOFF).sort(keys)
    after = tampered.filter(pl.col("origin_date") <= CUTOFF).sort(keys)

    assert not before.get_column("org_lag_1").equals(after.get_column("org_lag_1"))


def test_origin_is_exactly_horizon_days_before_target(
    training_frame: pl.DataFrame, feature_config: FeatureConfig
) -> None:
    """origin と target の間隔が常に horizon と一致すること。"""
    gaps = training_frame.select(
        (pl.col("date") - pl.col("origin_date")).dt.total_days().alias("gap"),
        pl.col("feat_horizon"),
    )
    mismatched = gaps.filter(pl.col("gap") != pl.col("feat_horizon"))
    assert mismatched.is_empty(), f"horizon と実際の間隔が食い違う行があります: {mismatched.head()}"
    assert gaps.get_column("feat_horizon").max() == feature_config.horizon


def test_training_and_inference_features_match(
    demand_frame: pl.DataFrame, feature_config: FeatureConfig
) -> None:
    """学習時と推論時で、同じ行に対して同じ特徴量が生成されること。

    training-serving skew（学習と推論で前処理がずれる事故）の検査。
    ここがずれると、オフラインの精度と本番の精度が一致しなくなる。
    """
    training = build_training_frame(demand_frame, feature_config)

    target_date = dt.date(2025, 5, 20)
    horizon = 5
    origin_date = target_date - dt.timedelta(days=horizon)
    store, sku = "S01", "SKU01"

    expected = training.filter(
        (pl.col("date") == target_date)
        & (pl.col("store_id") == store)
        & (pl.col("sku_id") == sku)
        & (pl.col("feat_horizon") == horizon)
    )
    assert expected.height == 1, "比較対象の学習行が一意に取れませんでした"

    series = demand_frame.filter((pl.col("store_id") == store) & (pl.col("sku_id") == sku))
    history = series.filter(pl.col("date") <= origin_date)
    future = series.filter(pl.col("date") == target_date).select(
        ["date", "store_id", "sku_id", "price", "promo_flag"]
    )

    actual = build_inference_frame(history, future, feature_config)
    assert actual.height == 1

    features = feature_columns(training)
    for column in features:
        expected_value = expected.get_column(column).item()
        actual_value = actual.get_column(column).item()
        if expected_value is None or actual_value is None:
            assert expected_value == actual_value, f"{column} の欠損状態が一致しません"
            continue
        assert actual_value == pytest.approx(expected_value), (
            f"特徴量 {column} が学習時と推論時で一致しません: "
            f"学習={expected_value}, 推論={actual_value}"
        )


def test_inference_rejects_out_of_range_horizon(
    demand_frame: pl.DataFrame, feature_config: FeatureConfig
) -> None:
    """horizon の範囲外を予測しようとしたら明示的に落ちること。"""
    series = demand_frame.filter((pl.col("store_id") == "S01") & (pl.col("sku_id") == "SKU01"))
    origin_date = dt.date(2025, 5, 1)
    history = series.filter(pl.col("date") <= origin_date)
    too_far = origin_date + dt.timedelta(days=feature_config.horizon + 1)
    future = series.filter(pl.col("date") == too_far).select(
        ["date", "store_id", "sku_id", "price", "promo_flag"]
    )

    with pytest.raises(ValueError, match="horizon が範囲外"):
        build_inference_frame(history, future, feature_config)
