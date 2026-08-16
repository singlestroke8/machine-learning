"""パイプライン全体のスモークテスト。

各部品の単体テストが通っていても、つなげたときに動くとは限らない。
CI では小さな設定でこのテストを回し、
「clone した人がコマンドを順に打てば動く」ことを保証する。

実行時間がかかるため ``slow`` マーカーを付けてある。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
import yaml

from demand_forecast.config import load_config
from demand_forecast.data.generate import generate_demand_data
from demand_forecast.data.loaders import write_frame
from demand_forecast.models.estimator import ForecastArtifact
from demand_forecast.models.train import run_training

pytestmark = pytest.mark.slow


@pytest.fixture
def smoke_config_path(tmp_path: Path) -> Path:
    """CI で数十秒で回りきる小さな設定を書き出す。"""
    config = {
        "seed": 42,
        "paths": {
            "raw": str(tmp_path / "data/raw/demand.parquet"),
            "processed": str(tmp_path / "data/processed/train.parquet"),
            "model_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
            "mlflow_tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        },
        "data": {
            "start_date": "2024-01-01",
            "end_date": "2025-06-30",
            "n_stores": 2,
            "n_skus": 2,
        },
        "features": {
            "horizon": 7,
            "lags": [1, 7],
            "rolling_windows": [7, 28],
            "fourier_yearly_order": 2,
        },
        "cv": {"n_splits": 2, "val_days": 14, "gap_days": 0},
        "model": {
            "quantiles": [0.1, 0.5, 0.9],
            "params": {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 15},
        },
        "tuning": {"n_trials": 2, "timeout_seconds": 60},
        "api": {"model_path": str(tmp_path / "models/model.joblib")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_generate_train_predict_end_to_end(smoke_config_path: Path) -> None:
    """データ生成 → 学習 → 保存 → 読み込み → 推論 が一気通貫で動くこと。"""
    cfg = load_config(smoke_config_path)

    # 1. データ生成
    demand = generate_demand_data(cfg.data, seed=cfg.seed)
    write_frame(demand, cfg.paths.raw)

    # 2. 学習（MLflow は使わない: CI を外部状態に依存させない）
    results = run_training(cfg, fast=False, track=False)

    assert results["n_train_rows"] > 0
    assert 0.0 < results["cv_summary"]["wape_mean"] < 1.0
    assert len(results["horizon_wape"]) == cfg.features.horizon

    # 3. 成果物が揃っていること
    reports_dir = Path(cfg.paths.reports_dir)
    for name in ("metrics.json", "feature_importance.csv", "model_card.md"):
        assert (reports_dir / name).exists(), f"{name} が生成されていません"

    metrics = json.loads((reports_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cv_summary"]["wape_mean"] == pytest.approx(results["cv_summary"]["wape_mean"])

    # 4. 保存済みモデルで推論できること
    artifact = ForecastArtifact.load(cfg.api.model_path)
    assert artifact.feature_config.horizon == cfg.features.horizon

    import polars as pl

    from demand_forecast.models.predict import forecast

    origin = demand.get_column("date").max()
    history = demand.filter((pl.col("store_id") == "S01") & (pl.col("sku_id") == "SKU01"))
    future = pl.DataFrame(
        {
            "date": [origin + dt.timedelta(days=h) for h in range(1, cfg.features.horizon + 1)],
            "store_id": ["S01"] * cfg.features.horizon,
            "sku_id": ["SKU01"] * cfg.features.horizon,
            "price": [400.0] * cfg.features.horizon,
            "promo_flag": [0] * cfg.features.horizon,
        },
        schema={
            "date": pl.Date,
            "store_id": pl.Utf8,
            "sku_id": pl.Utf8,
            "price": pl.Float64,
            "promo_flag": pl.Int8,
        },
    )
    predictions = forecast(artifact, history, future)
    assert predictions.height == cfg.features.horizon
    assert predictions.get_column("point").min() >= 0.0


def test_model_beats_every_baseline(smoke_config_path: Path) -> None:
    """学習したモデルが、すべての単純手法より良いこと。

    これはモデルの性能を保証するテストではなく、
    「特徴量の結合がずれてモデルが壊れた」ことを検知するための回帰テスト。
    どれか1つのベースラインにすら勝てないなら、まず実装を疑うべき状態にある。
    """
    cfg = load_config(smoke_config_path)
    write_frame(generate_demand_data(cfg.data, seed=cfg.seed), cfg.paths.raw)

    results = run_training(cfg, fast=False, track=False)
    model_wape = results["cv_summary"]["wape_mean"]

    baselines = {
        key.removesuffix("_wape_mean"): value
        for key, value in results["baseline_summary"].items()
        if key.endswith("_wape_mean")
    }
    assert baselines, "ベースラインが1つも評価されていません"
    for name, value in baselines.items():
        assert model_wape < value, (
            f"モデル({model_wape:.4f}) がベースライン {name}({value:.4f}) に負けています"
        )
