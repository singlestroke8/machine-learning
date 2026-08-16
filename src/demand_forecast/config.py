"""設定の型付き読み込み。

YAML をそのまま dict で持ち回すと、キーの打ち間違いが「実行して数分後に
KeyError」という形でしか表面化しない。pydantic モデルに載せることで、
起動直後に構造と型を検証できるようにしている。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("conf/config.yaml")


class PathConfig(BaseModel):
    """入出力パス。すべてリポジトリルートからの相対パスとして扱う。"""

    raw: Path = Path("data/raw/demand.parquet")
    processed: Path = Path("data/processed/train_frame.parquet")
    model_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"


class DataConfig(BaseModel):
    """合成データ生成の設定。"""

    start_date: dt.date
    end_date: dt.date
    n_stores: int = Field(ge=1)
    n_skus: int = Field(ge=1)

    @field_validator("end_date")
    @classmethod
    def _end_after_start(cls, v: dt.date, info: Any) -> dt.date:
        start = info.data.get("start_date")
        if start is not None and v <= start:
            msg = f"end_date ({v}) は start_date ({start}) より後である必要があります"
            raise ValueError(msg)
        return v


class FeatureConfig(BaseModel):
    """特徴量生成の設定。

    ``horizon`` は「何日先を予測するか」であると同時に、
    「特徴量が参照してよい情報の打ち切り位置」でもある。
    """

    horizon: int = Field(ge=1)
    lags: list[int]
    rolling_windows: list[int]
    fourier_yearly_order: int = Field(ge=0, le=10)

    @field_validator("lags", "rolling_windows")
    @classmethod
    def _positive_and_sorted(cls, v: list[int]) -> list[int]:
        if not v:
            msg = "少なくとも1つの値が必要です"
            raise ValueError(msg)
        if any(x < 1 for x in v):
            msg = f"すべて1以上である必要があります: {v}"
            raise ValueError(msg)
        return sorted(set(v))


class CVConfig(BaseModel):
    """時系列クロスバリデーションの設定。"""

    n_splits: int = Field(ge=1)
    val_days: int = Field(ge=1)
    gap_days: int = Field(ge=0)


class ModelConfig(BaseModel):
    """LightGBM 分位点回帰の設定。"""

    quantiles: list[float]
    params: dict[str, Any]

    @field_validator("quantiles")
    @classmethod
    def _valid_quantiles(cls, v: list[float]) -> list[float]:
        if any(not 0.0 < q < 1.0 for q in v):
            msg = f"分位点は 0 < q < 1 の範囲である必要があります: {v}"
            raise ValueError(msg)
        if 0.5 not in v:
            msg = "点予測に用いるため 0.5 を必ず含めてください"
            raise ValueError(msg)
        return sorted(v)


class TuningConfig(BaseModel):
    """Optuna によるハイパーパラメータ探索の設定。"""

    n_trials: int = Field(ge=1)
    timeout_seconds: int = Field(ge=1)


class ApiConfig(BaseModel):
    """推論APIの設定。"""

    model_path: Path = Path("models/model.joblib")


class Config(BaseModel):
    """設定ファイル全体。"""

    model_config = {"extra": "forbid"}

    seed: int
    paths: PathConfig
    data: DataConfig
    features: FeatureConfig
    cv: CVConfig
    model: ModelConfig
    tuning: TuningConfig
    api: ApiConfig


def load_config(path: Path | str | None = None) -> Config:
    """YAML から設定を読み込む。

    Args:
        path: 設定ファイルのパス。``None`` の場合は ``conf/config.yaml``。

    Raises:
        FileNotFoundError: 設定ファイルが存在しない場合。
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        msg = f"設定ファイルが見つかりません: {config_path}"
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return Config.model_validate(raw)


class ApiSettings(BaseSettings):
    """推論サーバの環境変数設定。

    コンテナ実行時は設定ファイルではなく環境変数で差し替えたいので、
    API 層だけは ``DFC_`` 接頭辞の環境変数から読む。
    """

    model_config = SettingsConfigDict(env_prefix="DFC_", extra="ignore")

    model_path: Path = Path("models/model.joblib")
    log_level: str = "INFO"
