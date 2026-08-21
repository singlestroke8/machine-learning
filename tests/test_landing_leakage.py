"""着地予測のリーク検査。

この課題で最も起きやすく、最も気づきにくい事故は
**k日目の予測に、k日目より後の受注を混ぜる**ことである。
エラーは出ず、検証の成績だけが良くなる。気づく手がかりがない。

そこで実装を読んで確かめるのではなく、**外から観測できる性質**で縛る。

    k日目より後の受注を10倍に改変しても、
    k日目の特徴量が1ビットも変わらないこと。

対になる逆向きの検査（k日目以前を変えたら特徴量は変わること）も置く。
これが無いと、「特徴量がそもそも計算されていない」場合にもテストが通ってしまう。
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from sales_analytics.data.generator import GeneratedData
from sales_analytics.tasks.landing.dataset import (
    GROUP_COL,
    MONTH_COL,
    STEP_COL,
    TARGET_COL,
    build_landing_frame,
    feature_columns,
)

#: 検査に使う対象月と経過日。前年同月の特徴量が揃う期間から選ぶ
TARGET_MONTH = dt.date(2024, 6, 1)
TARGET_STEP = 8


def _origin_date(transactions: pl.DataFrame, month: dt.date, step: int) -> dt.date:
    """対象月の k 営業日目が、暦のどの日かを求める。"""
    frame = build_landing_frame(transactions)
    row = frame.filter(
        (pl.col(GROUP_COL) == "全社") & (pl.col(MONTH_COL) == month) & (pl.col(STEP_COL) == step)
    )
    assert row.height == 1, f"対象の行が見つかりません: {month} の {step} 営業日目"
    days = (
        transactions.filter(pl.col("受注日").dt.truncate("1mo") == month)
        .get_column("受注日")
        .unique()
        .sort()
    )
    return days[step - 1]


def _features_at(frame: pl.DataFrame, month: dt.date, step: int) -> pl.DataFrame:
    return (
        frame.filter((pl.col(MONTH_COL) == month) & (pl.col(STEP_COL) == step))
        .sort(GROUP_COL)
        .select(feature_columns(frame))
    )


@pytest.fixture(scope="module")
def transactions(generated: GeneratedData) -> pl.DataFrame:
    return generated.transactions


def test_future_orders_do_not_change_current_features(transactions: pl.DataFrame) -> None:
    """k日目より後の受注を改変しても、k日目の特徴量が変わらないこと。

    このリポジトリで最も重要な検査。
    """
    origin = _origin_date(transactions, TARGET_MONTH, TARGET_STEP)
    before = _features_at(build_landing_frame(transactions), TARGET_MONTH, TARGET_STEP)

    tampered = transactions.with_columns(
        pl.when(pl.col("受注日") > origin)
        .then(pl.col("販売金額") * 10)
        .otherwise(pl.col("販売金額"))
        .alias("販売金額")
    )
    after = _features_at(build_landing_frame(tampered), TARGET_MONTH, TARGET_STEP)

    assert before.equals(after), "k日目より後の受注が、k日目の特徴量に影響しています"


def test_past_orders_do_change_current_features(transactions: pl.DataFrame) -> None:
    """逆向きの検査。k日目以前を変えたら、特徴量は変わること。

    これが無いと、特徴量が計算されていない場合にも上のテストが通ってしまう。
    """
    origin = _origin_date(transactions, TARGET_MONTH, TARGET_STEP)
    before = _features_at(build_landing_frame(transactions), TARGET_MONTH, TARGET_STEP)

    tampered = transactions.with_columns(
        pl.when(pl.col("受注日") <= origin)
        .then(pl.col("販売金額") * 10)
        .otherwise(pl.col("販売金額"))
        .alias("販売金額")
    )
    after = _features_at(build_landing_frame(tampered), TARGET_MONTH, TARGET_STEP)

    assert not before.equals(after), "過去を変えたのに特徴量が変わりません（未計算の疑い）"


def test_target_does_change_with_future_orders(transactions: pl.DataFrame) -> None:
    """目的変数は、k日目より後の受注で変わること。

    「着地額」は月全体の合計なので、変わるのが正しい。
    ここが変わらないなら、目的変数の作り方を間違えている。
    """
    origin = _origin_date(transactions, TARGET_MONTH, TARGET_STEP)
    tampered = transactions.with_columns(
        pl.when(pl.col("受注日") > origin)
        .then(pl.col("販売金額") * 10)
        .otherwise(pl.col("販売金額"))
        .alias("販売金額")
    )

    def target(frame: pl.DataFrame) -> float:
        row = frame.filter(
            (pl.col(GROUP_COL) == "全社")
            & (pl.col(MONTH_COL) == TARGET_MONTH)
            & (pl.col(STEP_COL) == TARGET_STEP)
        )
        return float(row.get_column(TARGET_COL).item())

    assert target(build_landing_frame(tampered)) > target(build_landing_frame(transactions))


def test_cumulative_never_exceeds_the_landing(transactions: pl.DataFrame) -> None:
    """当月の累計が、着地額を超えないこと。

    返品で一時的に累計が着地を上回ることはありうるが、
    最終営業日では必ず一致する。ここがずれていたら集計の作り方が壊れている。
    """
    frame = build_landing_frame(transactions)
    last_day = frame.filter(pl.col(STEP_COL) == pl.col("cal_月の営業日数"))
    assert (last_day.get_column("cum_金額") == last_day.get_column(TARGET_COL)).all()


def test_every_business_day_is_present(transactions: pl.DataFrame) -> None:
    """受注が1件も無かった営業日も、行として並んでいること。

    受注のあった日だけを並べると経過営業日 k がずれ、
    「10営業日目」と言いながら実際は12営業日目、ということが起きる。
    """
    frame = build_landing_frame(transactions)
    per_month = frame.group_by(GROUP_COL, MONTH_COL).agg(
        pl.len().alias("行数"), pl.col("cal_月の営業日数").first().alias("営業日数")
    )
    assert (per_month.get_column("行数") == per_month.get_column("営業日数")).all()


def test_features_exclude_the_target(transactions: pl.DataFrame) -> None:
    """目的変数そのものが特徴量に混ざっていないこと。"""
    frame = build_landing_frame(transactions)
    assert TARGET_COL not in feature_columns(frame)
    assert "金額" not in feature_columns(frame), "当日の実績（未来を含みうる列）が入っています"
