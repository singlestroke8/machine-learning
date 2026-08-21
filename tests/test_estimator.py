"""分位点回帰モデルと、その保存形式のテスト。

課題（回帰・分類・…）が変わっても使い回す部品なので、
特定の課題のデータ形式に依存しない小さな合成データで検証する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sales_analytics.models.encoding import UNKNOWN_CODE, CategoricalEncoder
from sales_analytics.models.estimator import ForecastArtifact, QuantileForecaster

FEATURES = ["feat_a", "feat_b", "feat_部署_code", "feat_品名_code"]


@pytest.fixture(scope="module")
def toy_frame() -> pl.DataFrame:
    """目的変数が特徴量から決まる、手で追える小さなデータ。"""
    rng = np.random.default_rng(0)
    n = 600
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(0.0, 1.0, n)
    return pl.DataFrame(
        {
            "部署": ["営業1部" if i % 2 else "営業2部" for i in range(n)],
            "品名": ["ノートPC_標準モデル" if i % 3 else "会計ソフト" for i in range(n)],
            "feat_a": a,
            "feat_b": b,
            "feat_部署_code": [i % 2 for i in range(n)],
            "feat_品名_code": [i % 3 for i in range(n)],
            "y": np.maximum(0.0, 10.0 + 3.0 * a - 2.0 * b + rng.normal(0.0, 1.0, n)),
        }
    )


@pytest.fixture(scope="module")
def trained(toy_frame: pl.DataFrame) -> ForecastArtifact:
    model = QuantileForecaster(
        quantiles=[0.1, 0.5, 0.9],
        params={"n_estimators": 40, "num_leaves": 15, "random_state": 0},
    )
    model.fit(toy_frame, FEATURES)
    return ForecastArtifact(
        model=model,
        encoder=CategoricalEncoder.fit(toy_frame, ["部署", "品名"]),
        metadata={"trained_at": "2026-01-01T00:00:00+00:00", "n_rows": toy_frame.height},
    )


def test_predictions_are_non_negative(trained: ForecastArtifact, toy_frame: pl.DataFrame) -> None:
    """予測が負にならないこと（数量・金額は負を取らない）。"""
    for values in trained.model.predict(toy_frame).values():
        assert values.min() >= 0.0


def test_quantiles_do_not_cross(trained: ForecastArtifact, toy_frame: pl.DataFrame) -> None:
    """分位点の大小関係が全行で保たれること（分位点交差の解消）。

    別々に学習した3本のモデルは、そのままだと下側が中央値を上回ることがある。
    順序が壊れると「下限のほうが上限より大きい区間」を返してしまう。
    """
    predictions = trained.model.predict(toy_frame)
    lower, median, upper = predictions[0.1], predictions[0.5], predictions[0.9]
    assert np.all(lower <= median)
    assert np.all(median <= upper)


def test_predict_before_fit_raises() -> None:
    model = QuantileForecaster(quantiles=[0.5], params={})
    with pytest.raises(RuntimeError, match="未学習"):
        model.predict(pl.DataFrame({"a": [1.0]}))


def test_missing_feature_raises(trained: ForecastArtifact, toy_frame: pl.DataFrame) -> None:
    """推論時に特徴量が欠けていたら、静かに間違えるのではなく落ちること。"""
    truncated = toy_frame.head(10).drop(trained.model.feature_names[0])
    with pytest.raises(KeyError, match="特徴量が不足"):
        trained.model.predict(truncated)


def test_feature_importance_is_sorted(trained: ForecastArtifact) -> None:
    importance = trained.model.feature_importance()
    values = importance.get_column("importance").to_list()
    assert values == sorted(values, reverse=True)
    assert importance.height == len(trained.model.feature_names)


def test_feature_importance_rejects_unknown_quantile(trained: ForecastArtifact) -> None:
    with pytest.raises(KeyError, match=r"0\.42"):
        trained.model.feature_importance(quantile=0.42)


def test_artifact_roundtrip(trained: ForecastArtifact, tmp_path: Path) -> None:
    """保存して読み直しても、メタデータとID対応表が保たれること。"""
    loaded = ForecastArtifact.load(trained.save(tmp_path / "model.joblib"))
    assert loaded.metadata == trained.metadata
    assert loaded.encoder.mappings == trained.encoder.mappings
    assert loaded.model.feature_names == trained.model.feature_names


def test_loading_missing_model_gives_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="学習"):
        ForecastArtifact.load(tmp_path / "absent.joblib")


def test_loading_wrong_format_raises(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "not_an_artifact.joblib"
    joblib.dump({"just": "a dict"}, path)
    with pytest.raises(ValueError, match="想定外の形式"):
        ForecastArtifact.load(path)


def test_artifact_version_mismatch_is_detected(trained: ForecastArtifact, tmp_path: Path) -> None:
    """保存形式が変わったモデルを読もうとしたら、再学習を促して落ちること。"""
    import copy

    import joblib

    stale = copy.copy(trained)
    stale.format_version = 999
    path = tmp_path / "stale.joblib"
    joblib.dump(stale, path)
    with pytest.raises(ValueError, match="再学習"):
        ForecastArtifact.load(path)


def test_unknown_series_ids_fall_back_to_missing_category(trained: ForecastArtifact) -> None:
    """学習時に無かったIDは -1（欠損カテゴリ）になること。

    新しい顧客や商品が来たときに落ちるのではなく、
    「知らないもの」として扱えないと運用に耐えない。
    """
    unknown = pl.DataFrame({"部署": ["新設部署"], "品名": ["会計ソフト"]})
    encoded = trained.encoder.transform(unknown)
    assert encoded.get_column("feat_部署_code").item() == UNKNOWN_CODE
    assert encoded.get_column("feat_品名_code").item() >= 0
