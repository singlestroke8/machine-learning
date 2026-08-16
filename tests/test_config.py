"""設定読み込みのテスト。

リポジトリに入っている本番設定そのものを検証対象に含めている。
設定ファイルの書き間違いは、学習を回して初めて気づくことが多く、
そのぶん時間を無駄にするため。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import yaml

from demand_forecast.config import (
    Config,
    FeatureConfig,
    ModelConfig,
    load_config,
)


def test_repo_config_is_valid(repo_config: Config) -> None:
    """conf/config.yaml が現在のスキーマで読み込めること。"""
    assert repo_config.features.horizon >= 1
    assert 0.5 in repo_config.model.quantiles


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="設定ファイルが見つかりません"):
        load_config(tmp_path / "nope.yaml")


def test_unknown_key_is_rejected(tmp_path: Path, repo_config: Config) -> None:
    """設定に知らないキーがあったら弾くこと（打ち間違いの検出）。"""
    payload = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "conf" / "config.yaml").read_text(encoding="utf-8")
    )
    payload["typo_key"] = 1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="typo_key"):
        load_config(path)


def test_end_date_must_follow_start_date() -> None:
    from demand_forecast.config import DataConfig

    with pytest.raises(ValueError, match="より後である必要があります"):
        DataConfig(
            start_date=dt.date(2026, 1, 1),
            end_date=dt.date(2025, 1, 1),
            n_stores=1,
            n_skus=1,
        )


def test_lags_are_deduplicated_and_sorted() -> None:
    cfg = FeatureConfig(
        horizon=7, lags=[7, 1, 1, 2], rolling_windows=[28, 7], fourier_yearly_order=1
    )
    assert cfg.lags == [1, 2, 7]
    assert cfg.rolling_windows == [7, 28]


def test_non_positive_lag_is_rejected() -> None:
    with pytest.raises(ValueError, match="1以上"):
        FeatureConfig(horizon=7, lags=[0, 1], rolling_windows=[7], fourier_yearly_order=1)


def test_quantiles_must_include_median() -> None:
    """点予測に使う 0.5 が無い設定を弾くこと。"""
    with pytest.raises(ValueError, match=r"0\.5"):
        ModelConfig(quantiles=[0.1, 0.9], params={})


def test_quantiles_must_be_within_open_unit_interval() -> None:
    with pytest.raises(ValueError, match="0 < q < 1"):
        ModelConfig(quantiles=[0.0, 0.5], params={})


def test_api_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from demand_forecast.config import ApiSettings

    monkeypatch.setenv("DFC_MODEL_PATH", "/tmp/some_model.joblib")
    monkeypatch.setenv("DFC_LOG_LEVEL", "DEBUG")
    settings = ApiSettings()
    assert settings.model_path == Path("/tmp/some_model.joblib")
    assert settings.log_level == "DEBUG"
