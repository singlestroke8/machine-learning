"""モデル本体と保存形式のテスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from demand_forecast.features.pipeline import feature_columns
from demand_forecast.models.estimator import ForecastArtifact, QuantileForecaster
from demand_forecast.models.predict import InsufficientHistoryError, forecast


def test_predictions_are_non_negative(
    trained_artifact: ForecastArtifact, training_frame: pl.DataFrame
) -> None:
    """需要予測が負にならないこと。"""
    predictions = trained_artifact.model.predict(training_frame.head(500))
    for values in predictions.values():
        assert values.min() >= 0.0


def test_quantiles_do_not_cross(
    trained_artifact: ForecastArtifact, training_frame: pl.DataFrame
) -> None:
    """分位点の大小関係が全行で保たれること（分位点交差の解消）。"""
    predictions = trained_artifact.model.predict(training_frame.head(1000))
    lower, median, upper = predictions[0.1], predictions[0.5], predictions[0.9]
    assert np.all(lower <= median)
    assert np.all(median <= upper)


def test_predict_before_fit_raises() -> None:
    model = QuantileForecaster(quantiles=[0.5], params={})
    with pytest.raises(RuntimeError, match="未学習"):
        model.predict(pl.DataFrame({"a": [1.0]}))


def test_missing_feature_raises(
    trained_artifact: ForecastArtifact, training_frame: pl.DataFrame
) -> None:
    """推論時に特徴量が欠けていたら、静かに間違えるのではなく落ちること。"""
    truncated = training_frame.head(10).drop(trained_artifact.model.feature_names[0])
    with pytest.raises(KeyError, match="特徴量が不足"):
        trained_artifact.model.predict(truncated)


def test_feature_importance_is_sorted(trained_artifact: ForecastArtifact) -> None:
    importance = trained_artifact.model.feature_importance()
    values = importance.get_column("importance").to_list()
    assert values == sorted(values, reverse=True)
    assert importance.height == len(trained_artifact.model.feature_names)


def test_feature_importance_rejects_unknown_quantile(trained_artifact: ForecastArtifact) -> None:
    with pytest.raises(KeyError, match=r"0\.42"):
        trained_artifact.model.feature_importance(quantile=0.42)


def test_artifact_roundtrip(trained_artifact: ForecastArtifact, tmp_path: Path) -> None:
    """保存して読み直しても、特徴量設定とID対応表が保たれること。"""
    path = trained_artifact.save(tmp_path / "model.joblib")
    loaded = ForecastArtifact.load(path)

    assert loaded.feature_config == trained_artifact.feature_config
    assert loaded.encoder.store_to_code == trained_artifact.encoder.store_to_code
    assert loaded.model.feature_names == trained_artifact.model.feature_names


def test_loading_missing_model_gives_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dfc train"):
        ForecastArtifact.load(tmp_path / "absent.joblib")


def test_loading_wrong_format_raises(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "not_an_artifact.joblib"
    joblib.dump({"just": "a dict"}, path)
    with pytest.raises(ValueError, match="想定外の形式"):
        ForecastArtifact.load(path)


def test_artifact_version_mismatch_is_detected(
    trained_artifact: ForecastArtifact, tmp_path: Path
) -> None:
    """保存形式が変わったモデルを読もうとしたら、再学習を促して落ちること。"""
    import copy

    import joblib

    stale = copy.copy(trained_artifact)
    stale.format_version = 999
    path = tmp_path / "stale.joblib"
    joblib.dump(stale, path)

    with pytest.raises(ValueError, match="再学習"):
        ForecastArtifact.load(path)


def test_unknown_series_ids_fall_back_to_missing_category(
    trained_artifact: ForecastArtifact,
) -> None:
    """学習時に無かった店舗IDは -1（欠損カテゴリ）になること。"""
    frame = pl.DataFrame({"store_id": ["UNKNOWN"], "sku_id": ["SKU01"]})
    encoded = trained_artifact.encoder.transform(frame)
    assert encoded.get_column("feat_store_code").item() == -1
    assert encoded.get_column("feat_sku_code").item() >= 0


def test_forecast_rejects_empty_history(trained_artifact: ForecastArtifact) -> None:
    empty = pl.DataFrame(
        schema={
            "date": pl.Date,
            "store_id": pl.Utf8,
            "sku_id": pl.Utf8,
            "units_sold": pl.Int32,
            "price": pl.Float64,
            "promo_flag": pl.Int8,
        }
    )
    with pytest.raises(InsufficientHistoryError, match="空です"):
        forecast(trained_artifact, empty, empty)


def test_forecast_rejects_ragged_history(
    trained_artifact: ForecastArtifact, demand_frame: pl.DataFrame
) -> None:
    """系列ごとに履歴の最終日が違う入力を弾くこと。"""
    import datetime as dt

    cutoff = dt.date(2025, 6, 1)
    ragged = pl.concat(
        [
            demand_frame.filter((pl.col("sku_id") == "SKU01") & (pl.col("date") <= cutoff)),
            demand_frame.filter(
                (pl.col("sku_id") == "SKU02") & (pl.col("date") <= cutoff - dt.timedelta(days=3))
            ),
        ]
    )
    future = ragged.head(1).select(["date", "store_id", "sku_id", "price", "promo_flag"])

    with pytest.raises(InsufficientHistoryError, match="最終日が異なります"):
        forecast(trained_artifact, ragged, future)


def test_all_features_are_numeric(training_frame: pl.DataFrame) -> None:
    """特徴量に文字列が混ざっていないこと（学習時に落ちるのを未然に防ぐ）。"""
    for column in feature_columns(training_frame):
        assert training_frame.schema[column].is_numeric(), f"{column} が数値型ではありません"
