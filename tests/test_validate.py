"""合格基準の検査のテスト。

検査そのものが壊れていると、不良なデータを通してしまう。
「基準を満たさないデータを、ちゃんと落とせるか」を確かめる。

**合格基準は本番の設定（200社・3年）に合わせて決めている。**
テスト用の小さなデータ（40社・2年）は、大型案件1件の影響が大きく出るため
前年同月比が本番より大きく振れる。小さいデータで基準を判定するのは誤りなので、
本番設定での検査は ``slow`` として分けている。
"""

from __future__ import annotations

import polars as pl
import pytest

from sales_analytics.config import Config
from sales_analytics.data.generator import GeneratedData, generate
from sales_analytics.data.validate import Check, format_checks, run_checks


@pytest.mark.slow
def test_production_config_passes_all_checks(repo_config: Config) -> None:
    """conf/config.yaml の設定で生成したデータが、全基準を満たすこと。

    生成器のパラメータを触ると、ここが最初に落ちる。
    """
    data = generate(repo_config.transactions, seed=repo_config.seed)
    checks = run_checks(data.transactions, data.anomaly_labels)
    failed = [c for c in checks if not c.passed]
    assert not failed, "\n" + format_checks(checks)


def test_check_detects_sell_below_cost(generated: GeneratedData) -> None:
    """原価割れを混ぜたら、検査が気づくこと。

    検査が常に OK を返すだけの飾りになっていないことの確認。
    """
    broken = generated.transactions.with_columns(
        pl.when(pl.int_range(pl.len()) < 50).then(-1000).otherwise(pl.col("粗利")).alias("粗利")
    )
    checks = run_checks(broken, generated.anomaly_labels)
    margin_check = next(c for c in checks if c.name == "通常取引の原価割れ")
    assert not margin_check.passed


def test_check_detects_too_regular_data(generated: GeneratedData) -> None:
    """規則的すぎるデータを弾けること。

    毎年まったく同じ売上のデータでは前年同月比が完璧に当たる。
    そのようなデータで機械学習をやっても意味がないので、基準で落とす。
    """
    regular = generated.transactions.with_columns(
        (pl.col("受注日").dt.month() * 1_000_000).alias("販売金額")
    )
    checks = run_checks(regular, generated.anomaly_labels)
    yoy = next(c for c in checks if c.name == "前年同月比の外し率")
    assert not yoy.passed, "毎年同じ売上のデータが基準を通ってしまいます"


def test_check_detects_missing_department_months(generated: GeneratedData) -> None:
    """部署×月に穴があったら気づくこと。月次の着地予測が成立しなくなる。"""
    reduced = generated.transactions.filter(
        ~((pl.col("部署") == pl.col("部署").first()) & (pl.col("受注日").dt.month() == 7))
    )
    checks = run_checks(reduced, generated.anomaly_labels)
    empty = next(c for c in checks if c.name == "受注の無い部署×月")
    assert not empty.passed


def test_format_shows_reason_only_for_failures() -> None:
    """通った項目に説明を並べて読みにくくしないこと。"""
    passing = format_checks([Check("項目A", "1.0", True, "説明は出ないはず")])
    assert "OK" in passing
    assert "→" not in passing

    failing = format_checks([Check("項目B", "9.9", False, "こう直す")])
    assert "NG" in failing
    assert "→ こう直す" in failing
