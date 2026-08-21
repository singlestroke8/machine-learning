"""設定の型付き読み込み。

YAML をそのまま dict で持ち回すと、キーの打ち間違いが「実行して数分後に
KeyError」という形でしか表面化しない。pydantic モデルに載せることで、
起動直後に構造と型を検証できるようにしている。

いまはデータ生成に必要な設定だけを持つ。学習の設定は、課題を実装する
時点で追加する。**使っていない設定項目を先回りして置かない**（何が
効いているのか読めなくなるため）。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_PATH = Path("conf/config.yaml")


class PathConfig(BaseModel):
    """入出力パス。すべてリポジトリルートからの相対パスとして扱う。"""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    model_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"


class TransactionsConfig(BaseModel):
    """取引明細（ローデータ）の生成設定。

    実務で受け取るのは集計前の明細なので、そこから作り直せるようにしている。
    ここに置くのは「業務設定として人が決める値」だけで、分布の形を決める
    定数は ``data.masters`` / ``data.generator`` 側に置く。
    """

    start_date: dt.date = dt.date(2022, 1, 1)
    end_date: dt.date = dt.date(2024, 12, 31)
    #: 顧客数。多いほど顧客単位の課題（離反予測・セグメント）が安定する
    n_customers: int = Field(default=200, ge=10)
    #: 部署あたりの営業担当者数
    reps_per_department: int = Field(default=5, ge=1)
    #: 期間の途中から取引が始まる顧客の割合（新規獲得）
    new_customer_ratio: float = Field(default=0.25, ge=0.0, le=0.6)
    #: 期間の途中で取引が先細って止まる顧客の割合（離反）
    churn_ratio: float = Field(default=0.18, ge=0.0, le=0.6)

    @field_validator("end_date")
    @classmethod
    def _end_after_start(cls, v: dt.date, info: Any) -> dt.date:
        start = info.data.get("start_date")
        if start is not None and v <= start:
            msg = f"end_date ({v}) は start_date ({start}) より後である必要があります"
            raise ValueError(msg)
        return v

    @field_validator("churn_ratio")
    @classmethod
    def _leaves_enough_customers(cls, v: float, info: Any) -> float:
        new_ratio = info.data.get("new_customer_ratio", 0.0)
        if v + new_ratio >= 0.9:
            msg = (
                f"新規 {new_ratio} + 離反 {v} が大きすぎます。"
                "期間を通じて取引のある顧客が残らなくなります"
            )
            raise ValueError(msg)
        return v


class Config(BaseModel):
    """設定ファイル全体。"""

    model_config = {"extra": "forbid"}

    seed: int
    paths: PathConfig = Field(default_factory=PathConfig)
    transactions: TransactionsConfig = Field(default_factory=TransactionsConfig)


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
