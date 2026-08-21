"""設定読み込みのテスト。

リポジトリに入っている本番設定そのものを検証対象に含めている。
設定ファイルの書き間違いは、実行して初めて気づくことが多く、
そのぶん時間を無駄にするため。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import yaml

from sales_analytics.config import Config, TransactionsConfig, load_config


def test_repo_config_is_valid(repo_config: Config) -> None:
    """conf/config.yaml が現在のスキーマで読み込めること。"""
    assert repo_config.transactions.n_customers >= 10
    assert repo_config.transactions.start_date < repo_config.transactions.end_date


def test_repo_config_covers_at_least_two_years(repo_config: Config) -> None:
    """年周期を学習するには、学習期間に年周期が2回以上必要になる。"""
    span_days = (repo_config.transactions.end_date - repo_config.transactions.start_date).days
    assert span_days >= 730, "期間が2年未満だと季節性を学習できない"


def test_missing_file_raises(tmp_path: Path) -> None:
    """設定ファイルが無ければ、その場で分かるように落ちること。"""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "no_such_file.yaml")


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """設定の書き間違いを黙って無視しないこと。

    未知のキーを許すと、``n_customers`` を ``n_customer`` と書いた場合に
    既定値のまま動いてしまい、原因を追うのが難しくなる。
    """
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"seed": 1, "unknown_section": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown_section"):
        load_config(path)


def test_end_date_must_be_after_start_date() -> None:
    """期間が逆転していたら、生成前に弾くこと。"""
    with pytest.raises(ValueError, match="end_date"):
        TransactionsConfig(start_date=dt.date(2024, 1, 1), end_date=dt.date(2023, 1, 1))


def test_churn_and_new_ratio_cannot_consume_everything() -> None:
    """新規と離反で顧客を食い尽くす設定を弾くこと。

    通しで取引のある顧客が居なくなると、比較の基準が無くなる。
    """
    with pytest.raises(ValueError, match="大きすぎます"):
        TransactionsConfig(new_customer_ratio=0.5, churn_ratio=0.5)
