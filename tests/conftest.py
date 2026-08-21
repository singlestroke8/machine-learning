"""テスト共通のフィクスチャ。

テストは小さな合成データで回す。実データや生成済みファイルに依存させると、
CI で落ちたときに「コードが壊れたのか環境が壊れたのか」の切り分けができない。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from sales_analytics.config import Config, TransactionsConfig, load_config
from sales_analytics.data.generator import GeneratedData, generate

#: テスト用の小さな設定。本番より短い期間・少ない顧客にして高速化する。
#: ただし期間は2年ぶん取る。前年同月の比較を要する検査があるため。
TEST_TRANSACTIONS_CONFIG = TransactionsConfig(
    start_date=dt.date(2023, 1, 1),
    end_date=dt.date(2024, 12, 31),
    n_customers=40,
    reps_per_department=2,
    new_customer_ratio=0.25,
    churn_ratio=0.20,
)


@pytest.fixture(scope="session")
def generated() -> GeneratedData:
    """テスト用に生成した取引明細一式。"""
    return generate(TEST_TRANSACTIONS_CONFIG, seed=7)


@pytest.fixture(scope="session")
def repo_config() -> Config:
    """リポジトリに入っている本番設定（設定ファイル自体の検証用）。"""
    return load_config(Path(__file__).resolve().parents[1] / "conf" / "config.yaml")
