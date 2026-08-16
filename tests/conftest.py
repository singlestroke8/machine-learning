"""テスト共通のフィクスチャ。

テストは小さな合成データで回す。実データや学習済みモデルに依存させると、
CI で落ちたときに「コードが壊れたのか環境が壊れたのか」の切り分けができない。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from demand_forecast.config import Config, DataConfig, FeatureConfig, load_config
from demand_forecast.data.generate import generate_demand_data
from demand_forecast.features.pipeline import (
    SeriesEncoder,
    build_training_frame,
    feature_columns,
)
from demand_forecast.models.estimator import ForecastArtifact, QuantileForecaster

# テスト用の小さな設定。本番設定より短い期間・少ない系列にして高速化する。
TEST_DATA_CONFIG = DataConfig(
    start_date=dt.date(2024, 1, 1),
    end_date=dt.date(2025, 6, 30),
    n_stores=2,
    n_skus=2,
)

TEST_FEATURE_CONFIG = FeatureConfig(
    horizon=7,
    lags=[1, 2, 7],
    rolling_windows=[7, 28],
    fourier_yearly_order=2,
)


@pytest.fixture(scope="session")
def demand_frame() -> pl.DataFrame:
    """テスト用の需要データ。"""
    return generate_demand_data(TEST_DATA_CONFIG, seed=7)


@pytest.fixture(scope="session")
def feature_config() -> FeatureConfig:
    return TEST_FEATURE_CONFIG


@pytest.fixture(scope="session")
def training_frame(demand_frame: pl.DataFrame) -> pl.DataFrame:
    """学習用の特徴量行列。"""
    encoder = SeriesEncoder.fit(demand_frame)
    return encoder.transform(build_training_frame(demand_frame, TEST_FEATURE_CONFIG))


@pytest.fixture(scope="session")
def trained_artifact(demand_frame: pl.DataFrame, training_frame: pl.DataFrame) -> ForecastArtifact:
    """小さく学習したモデル一式。API テストなどで使う。"""
    features = feature_columns(training_frame)
    model = QuantileForecaster(
        quantiles=[0.1, 0.5, 0.9],
        params={"n_estimators": 30, "num_leaves": 15, "random_state": 0},
    )
    model.fit(training_frame, features)
    return ForecastArtifact(
        model=model,
        encoder=SeriesEncoder.fit(demand_frame),
        feature_config=TEST_FEATURE_CONFIG,
        metadata={"trained_at": "2026-01-01T00:00:00+00:00", "n_series": 4},
    )


@pytest.fixture
def artifact_path(trained_artifact: ForecastArtifact, tmp_path: Path) -> Path:
    """学習済みモデルを一時ファイルに保存したパス。"""
    return trained_artifact.save(tmp_path / "model.joblib")


@pytest.fixture(scope="session")
def repo_config() -> Config:
    """リポジトリに入っている本番設定（設定ファイル自体の検証用）。"""
    return load_config(Path(__file__).resolve().parents[1] / "conf" / "config.yaml")
