"""ベースラインのテスト。

ベースラインは「モデルの良し悪しを測る物差し」なので、
ここが間違っていると改善率の主張そのものが崩れる。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from demand_forecast.models.baselines import BASELINE_NAMES, baseline_column, compute_baselines


def test_all_baselines_are_computable(training_frame: pl.DataFrame) -> None:
    predictions = compute_baselines(training_frame, window=7)
    assert set(predictions) == set(BASELINE_NAMES)
    for values in predictions.values():
        assert len(values) == training_frame.height


def test_baselines_are_non_negative_and_finite(training_frame: pl.DataFrame) -> None:
    """履歴不足の欠損を 0 で埋め、NaN を残さないこと。"""
    for name, values in compute_baselines(training_frame, window=7).items():
        assert values.min() >= 0.0, f"{name} に負の値があります"
        assert np.isfinite(values).all(), f"{name} に NaN/inf が残っています"


def test_seasonal_naive_uses_same_weekday(training_frame: pl.DataFrame) -> None:
    """季節性ナイーブが、target と同じ曜日の実績を参照していること。"""
    predictions = compute_baselines(training_frame, window=7)
    scored = training_frame.select(["date", "org_target_dow_last"]).with_columns(
        pl.Series("baseline", predictions["seasonal_naive"])
    )
    mismatched = scored.filter(
        pl.col("org_target_dow_last").is_not_null()
        & (pl.col("org_target_dow_last") != pl.col("baseline"))
    )
    assert mismatched.is_empty()


def test_unknown_baseline_name_raises() -> None:
    with pytest.raises(KeyError, match="未知のベースライン"):
        baseline_column("magic", window=7)


def test_missing_column_is_skipped(training_frame: pl.DataFrame) -> None:
    """参照カラムが存在しない窓幅を指定しても落ちないこと。"""
    predictions = compute_baselines(training_frame, window=999)
    assert "moving_average" not in predictions
    assert "naive" in predictions
