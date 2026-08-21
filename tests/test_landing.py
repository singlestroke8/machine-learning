"""着地予測の学習データとベースラインのテスト。

リーク検査は ``test_landing_leakage.py`` に分けてある。
こちらは「予測の単位が意図どおりに組み上がっているか」を見る。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sales_analytics.data.generator import GeneratedData
from sales_analytics.tasks.landing.baselines import BASELINES, add_baselines, baseline_columns
from sales_analytics.tasks.landing.dataset import (
    ANCHOR_COL,
    COMPANY_LABEL,
    GROUP_COL,
    MONTH_COL,
    STEP_COL,
    TARGET_COL,
    build_landing_frame,
    feature_columns,
)


@pytest.fixture(scope="module")
def landing(generated: GeneratedData) -> pl.DataFrame:
    return add_baselines(build_landing_frame(generated.transactions))


# --- 予測の単位 -------------------------------------------------------------


def test_one_row_per_group_month_step(landing: pl.DataFrame) -> None:
    """1行 = （部署, 対象月, 経過営業日）であること。"""
    keys = landing.select(GROUP_COL, MONTH_COL, STEP_COL)
    assert keys.unique().height == landing.height


def test_company_is_the_sum_of_departments(landing: pl.DataFrame) -> None:
    """全社の着地額が、部署の合計と一致すること。"""
    monthly = landing.filter(pl.col(STEP_COL) == 1).select(GROUP_COL, MONTH_COL, TARGET_COL)
    company = monthly.filter(pl.col(GROUP_COL) == COMPANY_LABEL).sort(MONTH_COL)
    departments = (
        monthly.filter(pl.col(GROUP_COL) != COMPANY_LABEL)
        .group_by(MONTH_COL)
        .agg(pl.col(TARGET_COL).sum())
        .sort(MONTH_COL)
    )
    assert company.get_column(TARGET_COL).to_list() == departments.get_column(TARGET_COL).to_list()


def test_step_starts_at_one_and_reaches_the_month_length(landing: pl.DataFrame) -> None:
    per_month = landing.group_by(GROUP_COL, MONTH_COL).agg(
        pl.col(STEP_COL).min().alias("最小"),
        pl.col(STEP_COL).max().alias("最大"),
        pl.col("cal_月の営業日数").first().alias("営業日数"),
    )
    assert (per_month.get_column("最小") == 1).all()
    assert (per_month.get_column("最大") == per_month.get_column("営業日数")).all()


# --- 特徴量の規約 -----------------------------------------------------------


def test_features_are_scale_free(landing: pl.DataFrame) -> None:
    """特徴量に「円」の列が混ざっていないこと。

    金額のままだと系列ごとに桁が違い（部署間で約20倍）、さらに年々水準が
    上がるため、決定木が学習範囲の外を外挿できない。実際にそれで失敗した。
    """
    features = feature_columns(landing)
    yen_like = [c for c in features if c.startswith(("cum_", "prev_"))]
    assert not yen_like, f"金額のままの列が特徴量に入っています: {yen_like}"


def test_features_exclude_baselines(landing: pl.DataFrame) -> None:
    """ベースラインの予測値そのものが特徴量に入っていないこと。

    入れてしまうと「ベースラインに勝った」の意味が変わる。
    """
    assert not set(feature_columns(landing)) & set(baseline_columns())


# --- ベースライン -----------------------------------------------------------


def test_year_over_year_baseline_ignores_elapsed_days(landing: pl.DataFrame) -> None:
    """前年同月ベースラインが、経過日数によって変わらないこと。

    月初に出しても月末に出しても同じ数字を返す。これがこの方法の限界であり、
    機械学習を使う理由でもある。
    """
    one_month = landing.filter(
        (pl.col(GROUP_COL) == COMPANY_LABEL)
        & (pl.col(MONTH_COL) == landing.get_column(MONTH_COL).max())
    )
    values = one_month.get_column("base_前年同月").drop_nulls().unique()
    assert values.len() == 1, "経過日数で値が変わっています"


def test_remaining_split_equals_actual_on_the_last_day(landing: pl.DataFrame) -> None:
    """残り按分ベースラインが、最終営業日には累計と一致すること。

    最終営業日は残りがゼロなので、見込み額は確定した累計そのものになる。
    """
    last = landing.filter(pl.col(STEP_COL) == pl.col("cal_月の営業日数")).drop_nulls(
        "base_残り按分"
    )
    assert last.height > 0
    difference = (last.get_column("base_残り按分") - last.get_column("cum_金額")).abs()
    assert difference.max() < 1.0


def test_all_baselines_produce_values(landing: pl.DataFrame) -> None:
    usable = landing.filter(pl.col(ANCHOR_COL).is_not_null())
    for baseline in BASELINES:
        assert usable.get_column(baseline.column).drop_nulls().len() > 0, baseline.name


# --- 学習が通ること ---------------------------------------------------------


@pytest.mark.slow
def test_training_runs_and_reports_baselines(generated: GeneratedData, tmp_path: Path) -> None:
    """学習が最後まで通り、ベースライン比較が出ること。

    **勝つことは要求しない。** 勝てなかったという結果も成果なので、
    テストで「勝て」と縛ると、都合の良い検証条件を選ぶ動機が生まれる。
    """
    from sales_analytics.tasks.landing.train import format_summary, run_training

    summary = run_training(generated.transactions, reports_dir=tmp_path)
    assert summary["n_val_rows"] > 0
    assert set(summary["baselines"]) == {b.name for b in BASELINES}
    assert 0.0 <= summary["model"]["coverage"] <= 1.0
    assert (tmp_path / "landing_metrics.json").exists()
    assert "経過営業日ごとの WAPE" in format_summary(summary)
