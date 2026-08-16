"""営業日を時間軸にした場合の検査。

暦日データで通っているリーク検査が、営業日データでも同じように成立するかを見る。
時間軸を変えたときに壊れやすいのは次の3点なので、そこを重点的に検査する。

1. origin と target の結合が「horizon ステップぶん」ずれているか
   （暦日の加算で書くと、土日を跨いだ瞬間に対応する行が消える）
2. 同一曜日の参照が本当に同じ曜日を指しているか
   （営業日軸では1週間が5ステップなので、周期を7のままにするとずれる）
3. 学習時と推論時で同じ特徴量が出るか
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from demand_forecast.config import FeatureConfig, TransactionsConfig
from demand_forecast.data.aggregate import aggregate_transactions
from demand_forecast.data.transactions import generate_transactions
from demand_forecast.features.calendar import business_days
from demand_forecast.features.pipeline import (
    build_inference_frame,
    build_training_frame,
    feature_columns,
)
from demand_forecast.features.steps import build_timeline
from demand_forecast.models.splits import expanding_window_folds

BUSINESS_FEATURES = FeatureConfig(
    horizon=5,
    lags=[1, 2, 5],
    rolling_windows=[5, 20],
    fourier_yearly_order=2,
    calendar="business",
)

_SMALL_TRANSACTIONS = TransactionsConfig(
    start_date=dt.date(2023, 1, 1),
    end_date=dt.date(2023, 12, 31),
    n_customers=12,
    reps_per_department=2,
)


@pytest.fixture(scope="module")
def demand() -> pl.DataFrame:
    """営業日ベースに集計した需要データ。"""
    return aggregate_transactions(generate_transactions(_SMALL_TRANSACTIONS, seed=11))


# --- 時間軸そのもの -------------------------------------------------------


def test_timeline_contains_only_business_days(demand: pl.DataFrame) -> None:
    """集計結果に土日祝が含まれないこと。"""
    weekdays = demand.get_column("date").dt.weekday().unique().to_list()
    assert max(weekdays) <= 5

    dates = sorted(set(demand.get_column("date").to_list()))
    assert dates == business_days(dates[0], dates[-1])


def test_every_series_covers_every_business_day(demand: pl.DataFrame) -> None:
    """全系列が同じ営業日を持つこと（受注が無かった日は0で埋まっている）。"""
    per_series = demand.group_by(["store_id", "sku_id"]).agg(pl.len().alias("n"))
    assert per_series.get_column("n").n_unique() == 1
    assert demand.filter(pl.col("units_sold") == 0).height > 0


def test_build_timeline_skips_weekends() -> None:
    """時間軸の生成が土日を飛ばすこと。"""
    timeline = build_timeline(dt.date(2024, 1, 4), dt.date(2024, 1, 10), "business")
    assert dt.date(2024, 1, 6) not in timeline  # 土曜
    assert dt.date(2024, 1, 7) not in timeline  # 日曜
    assert dt.date(2024, 1, 8) not in timeline  # 成人の日
    assert timeline == [dt.date(2024, 1, d) for d in (4, 5, 9, 10)]


# --- 結合のずれ -----------------------------------------------------------


def test_origin_is_exactly_horizon_business_days_before_target(demand: pl.DataFrame) -> None:
    """origin と target の間隔が、暦日ではなく**営業日**で horizon と一致すること。

    暦日で結合していると、週をまたぐ行がそもそも作られず落ちる。
    ここは営業日で数え直して検査する。
    """
    frame = build_training_frame(demand, BUSINESS_FEATURES)
    timeline = build_timeline(
        *(lambda d: (d[0], d[-1]))(sorted(set(demand.get_column("date").to_list()))),
        "business",
    )
    step_of = {date: index for index, date in enumerate(timeline)}

    rows = frame.select(["date", "origin_date", "feat_horizon"]).unique().to_dicts()
    assert rows, "学習行が生成されていません"
    for row in rows:
        gap = step_of[row["date"]] - step_of[row["origin_date"]]
        assert gap == row["feat_horizon"], (
            f"営業日で数えた間隔 {gap} が horizon {row['feat_horizon']} と一致しません"
        )


def test_all_horizons_are_present(demand: pl.DataFrame) -> None:
    """週をまたぐ horizon の行も落ちずに残ること。"""
    frame = build_training_frame(demand, BUSINESS_FEATURES)
    horizons = sorted(frame.get_column("feat_horizon").unique().to_list())
    assert horizons == list(range(1, BUSINESS_FEATURES.horizon + 1))


# --- 曜日周期 -------------------------------------------------------------


def test_same_dow_reference_lands_on_same_weekday(demand: pl.DataFrame) -> None:
    """同一曜日の参照が、祝日を挟んでも本当に同じ曜日を指すこと。

    horizon から固定オフセットを計算する方式（「1週間 = 5営業日」）では、
    祝日のある週だけ間隔が縮んで曜日がずれる。曜日そのものを見て選ぶ方式なら
    ずれないので、実データで確認する。
    """
    frame = build_training_frame(demand, BUSINESS_FEATURES)

    # 参照した値が、実際にその曜日の実績と一致するかを突き合わせる
    actuals = demand.select(
        pl.col("date").alias("_ref_date"),
        pl.col("store_id"),
        pl.col("sku_id"),
        pl.col("units_sold").cast(pl.Float64).alias("_ref_value"),
    )
    checked = (
        frame.filter(pl.col("org_target_dow_last").is_not_null())
        .select(["date", "store_id", "sku_id", "origin_date", "org_target_dow_last"])
        .join(actuals, on=["store_id", "sku_id"], how="inner")
        .filter(
            (pl.col("_ref_date") <= pl.col("origin_date"))
            & (pl.col("_ref_date").dt.weekday() == pl.col("date").dt.weekday())
        )
        .group_by(["date", "store_id", "sku_id", "origin_date", "org_target_dow_last"])
        .agg(pl.col("_ref_value").sort_by("_ref_date").last().alias("expected"))
    )

    assert checked.height > 0
    mismatched = checked.filter(pl.col("org_target_dow_last") != pl.col("expected"))
    assert mismatched.is_empty(), (
        f"同一曜日の参照が {mismatched.height} 行でずれています: {mismatched.head(3).to_dicts()}"
    )


def test_calendar_mode_is_recorded() -> None:
    assert BUSINESS_FEATURES.calendar == "business"
    assert (
        FeatureConfig(horizon=5, lags=[1], rolling_windows=[5], fourier_yearly_order=1).calendar
        == "daily"
    )


# --- リーク防止（暦日と同じ性質が営業日でも成り立つか） -------------------


def test_future_actuals_do_not_change_past_features(demand: pl.DataFrame) -> None:
    """未来の実績を改変しても、origin が cutoff 以前の行が変わらないこと。"""
    cutoff = dt.date(2023, 9, 29)
    original = build_training_frame(demand, BUSINESS_FEATURES)
    tampered = build_training_frame(
        demand.with_columns(
            pl.when(pl.col("date") > cutoff)
            .then(pl.col("units_sold") * 10)
            .otherwise(pl.col("units_sold"))
            .cast(pl.Int32)
            .alias("units_sold")
        ),
        BUSINESS_FEATURES,
    )

    keys = ["date", "store_id", "sku_id", "feat_horizon"]
    features = [c for c in feature_columns(original) if c not in keys]
    before = original.filter(pl.col("origin_date") <= cutoff).sort(keys).select([*keys, *features])
    after = tampered.filter(pl.col("origin_date") <= cutoff).sort(keys).select([*keys, *features])

    assert before.height > 0
    assert before.equals(after), "営業日軸で未来の情報が特徴量に混入しています"


def test_training_and_inference_features_match(demand: pl.DataFrame) -> None:
    """営業日軸でも、学習時と推論時で同じ特徴量が出ること。"""
    training = build_training_frame(demand, BUSINESS_FEATURES)

    timeline = sorted(set(demand.get_column("date").to_list()))
    target_date = timeline[-1]
    horizon = 3
    origin_date = timeline[-1 - horizon]
    store = demand.get_column("store_id").first()
    sku = demand.get_column("sku_id").first()

    expected = training.filter(
        (pl.col("date") == target_date)
        & (pl.col("store_id") == store)
        & (pl.col("sku_id") == sku)
        & (pl.col("feat_horizon") == horizon)
    )
    assert expected.height == 1

    series = demand.filter((pl.col("store_id") == store) & (pl.col("sku_id") == sku))
    actual = build_inference_frame(
        series.filter(pl.col("date") <= origin_date),
        series.filter(pl.col("date") == target_date).select(
            ["date", "store_id", "sku_id", "price", "promo_flag"]
        ),
        BUSINESS_FEATURES,
    )
    assert actual.height == 1

    for column in feature_columns(training):
        expected_value = expected.get_column(column).item()
        actual_value = actual.get_column(column).item()
        if expected_value is None or actual_value is None:
            assert expected_value == actual_value, f"{column} の欠損状態が一致しません"
            continue
        assert actual_value == pytest.approx(expected_value), f"{column} が一致しません"


def test_inference_horizon_is_counted_in_business_days(demand: pl.DataFrame) -> None:
    """飛び飛びの予測対象日を渡しても、horizon が営業日で正しく数えられること。

    入力に現れる日付だけで連番を振っていると、ここが崩れる。
    """
    timeline = sorted(set(demand.get_column("date").to_list()))
    origin_date = timeline[-6]
    store = demand.get_column("store_id").first()
    sku = demand.get_column("sku_id").first()
    series = demand.filter((pl.col("store_id") == store) & (pl.col("sku_id") == sku))

    # 2営業日先と5営業日先だけを要求する（間を飛ばす）
    wanted = [timeline[-4], timeline[-1]]
    frame = build_inference_frame(
        series.filter(pl.col("date") <= origin_date),
        series.filter(pl.col("date").is_in(wanted)).select(
            ["date", "store_id", "sku_id", "price", "promo_flag"]
        ),
        BUSINESS_FEATURES,
    )
    assert sorted(frame.get_column("feat_horizon").to_list()) == [2, 5]


def test_inference_rejects_non_business_day(demand: pl.DataFrame) -> None:
    """営業日軸のモデルに土日の予測を求めたら弾かれること。"""
    timeline = sorted(set(demand.get_column("date").to_list()))
    origin_date = timeline[-5]
    store = demand.get_column("store_id").first()
    sku = demand.get_column("sku_id").first()
    series = demand.filter((pl.col("store_id") == store) & (pl.col("sku_id") == sku))

    saturday = origin_date + dt.timedelta(days=(5 - origin_date.weekday()) % 7 or 7)
    assert saturday.weekday() == 5

    future = pl.DataFrame(
        {
            "date": [saturday],
            "store_id": [store],
            "sku_id": [sku],
            "price": [100000.0],
            "promo_flag": [0],
        },
        schema={
            "date": pl.Date,
            "store_id": pl.Utf8,
            "sku_id": pl.Utf8,
            "price": pl.Float64,
            "promo_flag": pl.Int8,
        },
    )
    with pytest.raises(ValueError, match="ステップ対応表に無い日付"):
        build_inference_frame(
            series.filter(pl.col("date") <= origin_date), future, BUSINESS_FEATURES
        )


# --- CV -------------------------------------------------------------------


def test_cv_counts_validation_in_business_days(demand: pl.DataFrame) -> None:
    """検証期間が「20営業日」として数えられること（暦日28日ぶんではない）。"""
    dates = demand.get_column("date")
    folds = expanding_window_folds(dates, n_splits=3, val_steps=20)

    timeline = sorted(set(dates.to_list()))
    for fold in folds:
        span = [d for d in timeline if fold.val_start <= d <= fold.val_end]
        assert len(span) == 20, f"{fold.describe()} の検証期間が20営業日になっていません"
        assert fold.train_end < fold.val_start


# --- CLI の予測対象日 -----------------------------------------------------


def test_cli_future_dates_skip_holidays() -> None:
    """CLI が作る予測対象日が、土日祝を飛ばした営業日になること。

    暦日で ``origin + h 日`` として作ると休日に当たり、
    ステップ対応表に無い日付として推論が落ちる。
    """
    from demand_forecast.cli import _future_dates

    # 2024-07-12 は金曜。翌営業日は月曜 7/15 …ではなく、海の日なので 7/16（火）。
    origin = dt.date(2024, 7, 12)
    dates = _future_dates(origin, 3, "business")

    assert dates == [dt.date(2024, 7, 16), dt.date(2024, 7, 17), dt.date(2024, 7, 18)]
    assert dates == business_days(origin + dt.timedelta(days=1), dates[-1])


def test_cli_future_dates_are_calendar_days_in_daily_mode() -> None:
    """暦日軸では、休日を飛ばさず素直に翌日から並べること。"""
    from demand_forecast.cli import _future_dates

    origin = dt.date(2024, 7, 12)
    assert _future_dates(origin, 3, "daily") == [
        dt.date(2024, 7, 13),
        dt.date(2024, 7, 14),
        dt.date(2024, 7, 15),
    ]
