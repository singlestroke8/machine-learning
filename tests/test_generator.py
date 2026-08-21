"""取引明細の生成器のテスト。

生成器のテストで大事なのは「エラーなく動くこと」ではなく、
**業務ルールがデータに正しく反映されているか**である。
ここが崩れたデータで学習しても、モデルは業務と無関係なものを覚える。

前身のコードでは「金額の計算が合っているか」は検査していたが、
**符号を見ていなかった**ため、原価割れの受注が3%混入していた。
計算が合っていることと、業務的にありうる値であることは別の検査が要る。
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from sales_analytics.config import TransactionsConfig
from sales_analytics.data.generator import ANOMALY_KINDS, GeneratedData, generate
from sales_analytics.data.masters import (
    DEPARTMENTS,
    PRODUCT_INDEX,
    PRODUCTS,
    build_sales_reps,
    max_discount_for,
)

# --- 業務ルール -------------------------------------------------------------


def test_amounts_are_derived_from_quantity_and_price(generated: GeneratedData) -> None:
    """金額が数量と単価から導出されていること。"""
    t = generated.transactions
    assert (t.get_column("販売金額") == t.get_column("販売単価") * t.get_column("数量")).all()
    assert (t.get_column("原価") == t.get_column("原価単価") * t.get_column("数量")).all()
    assert (t.get_column("粗利") == t.get_column("販売金額") - t.get_column("原価")).all()


def test_normal_orders_never_sell_below_cost(generated: GeneratedData) -> None:
    """通常の受注に原価割れが無いこと。

    値引率の異常（``discount_error``）は**わざと**原価割れにしているので除く。
    それ以外で粗利が負になっていたら、値引率の上限が原価率を見ていない。
    """
    labelled = generated.anomaly_labels.select("受注番号", "明細番号").unique()
    normal = generated.transactions.join(labelled, on=["受注番号", "明細番号"], how="anti")
    negative = normal.filter((pl.col("数量") > 0) & (pl.col("粗利") < 0))
    assert negative.height == 0, f"原価割れが {negative.height} 行あります"


def test_discount_ceiling_differs_by_product() -> None:
    """原価率の高い商材ほど値引き余地が小さいこと。"""
    pc = PRODUCTS[PRODUCT_INDEX["デスクトップPC_標準モデル"]]
    software = PRODUCTS[PRODUCT_INDEX["業務管理ソフト"]]
    assert max_discount_for(pc) < max_discount_for(software)


def test_orders_fall_on_business_days_only(generated: GeneratedData) -> None:
    """土日に受注が発生していないこと（BtoB の前提）。"""
    weekdays = generated.transactions.get_column("受注日").dt.weekday().unique().to_list()
    assert max(weekdays) <= 5, "土日の受注が含まれています"


def test_each_rep_belongs_to_one_department(generated: GeneratedData) -> None:
    """1人の営業担当が複数部署に現れないこと。"""
    pairs = generated.transactions.select("営業担当者", "部署").unique()
    assert pairs.height == pairs.get_column("営業担当者").n_unique()


def test_departments_are_as_specified(generated: GeneratedData) -> None:
    assert set(generated.transactions.get_column("部署").unique()) <= set(DEPARTMENTS)


def test_too_many_reps_is_rejected() -> None:
    """名前の在庫を超える人数を要求したら、その場で落ちること。"""
    with pytest.raises(ValueError, match="多すぎます"):
        build_sales_reps(reps_per_department=50)


# --- 課題を成立させる性質 ---------------------------------------------------


def test_some_customers_start_late_and_some_stop(generated: GeneratedData) -> None:
    """顧客の出入りがあること。

    全員が期首から期末まで居るデータでは、離反予測が成立しない。
    """
    t = generated.transactions
    start, end = t.get_column("受注日").min(), t.get_column("受注日").max()
    span = t.group_by("顧客コード").agg(
        pl.col("受注日").min().alias("初回"), pl.col("受注日").max().alias("最終")
    )
    late = span.filter(pl.col("初回") > pl.lit(start) + pl.duration(days=90)).height
    stopped = span.filter(pl.col("最終") < pl.lit(end) - pl.duration(days=120)).height
    assert late > 0, "途中から取引が始まる顧客が居ません"
    assert stopped > 0, "途中で取引が途切れる顧客が居ません"


def test_churn_shows_multiple_weak_signals(generated: GeneratedData) -> None:
    """離反の予兆が、単一の列では説明できない形で現れること。

    2つの落とし穴があり、どちらも最初に踏んだ。

    1. 離反顧客と継続顧客をそのまま比べると、**規模の交絡**が入る。
       小さい顧客ほど値引率が低く、かつ離反しやすいので、予兆と逆の結論が出る。
       → 同じ顧客の中で、末期とそれ以前を比べる。
    2. 生の値引率は**商材構成に引きずられる**。ソフトは35%まで値引けるが
       PC は11%程度しか値引けないので、構成が変わるだけで平均が動く。
       → 商材ごとの値引き上限に対する比（交渉圧）で見る。
    """
    ceilings = {p.name: max_discount_for(p) for p in PRODUCTS}
    t = generated.transactions.filter(pl.col("数量") > 0)
    end = t.get_column("受注日").max()
    last = t.group_by("顧客コード").agg(pl.col("受注日").max().alias("最終"))
    churned = last.filter(pl.col("最終") < pl.lit(end) - pl.duration(days=120))
    assert churned.height > 0, "離反顧客が居ません"

    target = (
        t.join(churned, on="顧客コード", how="inner")
        .with_columns(
            (pl.col("受注日") > pl.col("最終") - pl.duration(days=90)).alias("末期"),
            pl.col("品名").replace_strict(ceilings).alias("値引き上限"),
        )
        .with_columns((pl.col("値引率") / pl.col("値引き上限")).alias("交渉圧"))
    )
    summary = target.group_by("末期").agg(
        pl.col("交渉圧").mean().alias("交渉圧"), pl.col("数量").mean().alias("数量")
    )
    late = summary.filter(pl.col("末期")).row(0, named=True)
    early = summary.filter(~pl.col("末期")).row(0, named=True)

    assert late["交渉圧"] > early["交渉圧"], "末期に値引き要求が強まっていません"
    assert late["数量"] < early["数量"], "末期に1回あたりが小さくなっていません"


def test_follow_up_purchases_create_product_association(generated: GeneratedData) -> None:
    """同時購入の組合せに偏りがあること（推薦の課題が成立する条件）。"""
    orders = (
        generated.transactions.filter(pl.col("数量") > 0)
        .group_by("受注番号")
        .agg(pl.col("品名").sort().alias("組"))
        .filter(pl.col("組").list.len() == 2)
        .with_columns(pl.col("組").list.join(" + ").alias("組合せ"))
    )
    counts = orders.get_column("組合せ").value_counts().sort("count", descending=True)
    assert counts.height >= 10
    top, bottom = counts.get_column("count")[0], counts.get_column("count")[-1]
    assert top >= bottom * 5, "商品の組合せが均等すぎて、推薦の学習に意味がありません"


def test_year_over_year_is_not_trivially_predictable(generated: GeneratedData) -> None:
    """前年同月比が当たりすぎないこと。

    毎年同じ月係数を使うと前年同月比が5%程度まで当たってしまい、
    「Excel で足りる」データになる。年ごとの揺らぎが効いているかを見る。
    """
    from sales_analytics.data.validate import _yoy_wape

    wape, n_months = _yoy_wape(generated.transactions)
    assert n_months > 0
    assert wape > 0.08, f"前年同月比の外し率が {wape:.1%} しかなく、規則的すぎます"


# --- 異常ラベル -------------------------------------------------------------


def test_anomaly_labels_reference_real_rows(generated: GeneratedData) -> None:
    """ラベルが実在する明細を指していること。"""
    matched = generated.anomaly_labels.join(
        generated.transactions.select("受注番号", "明細番号"),
        on=["受注番号", "明細番号"],
        how="semi",
    )
    assert matched.height == generated.anomaly_labels.height


def test_all_anomaly_kinds_are_present(generated: GeneratedData) -> None:
    kinds = set(generated.anomaly_labels.get_column("異常種別").unique())
    assert kinds == set(ANOMALY_KINDS)


def test_negative_quantity_is_not_the_same_as_anomaly(generated: GeneratedData) -> None:
    """「数量が負なら異常」という素朴な検知が誤検知すること。

    通常の返品と、押し込みの取消が混ざっている。この区別が付かないデータだと
    異常検知の課題が「符号を見るだけ」になってしまう。
    """
    negative = generated.transactions.filter(pl.col("数量") < 0)
    assert negative.height > 0
    flagged = negative.join(
        generated.anomaly_labels.select("受注番号", "明細番号").unique(),
        on=["受注番号", "明細番号"],
        how="semi",
    )
    assert flagged.height < negative.height, "数量が負の行がすべて異常になっています"


def test_customers_master_does_not_leak_the_answer(generated: GeneratedData) -> None:
    """顧客マスタに離反の正解が含まれていないこと。

    ``churn_start`` をマスタに書くと、離反予測の特徴量にそのまま使えてしまう。
    業務的にも「いつ離れるか」がマスタに載っていることはない。
    """
    columns = set(generated.customers.columns)
    assert "離反" not in columns
    assert not any("churn" in c.lower() for c in columns)


# --- 再現性 -----------------------------------------------------------------


def test_same_seed_gives_same_data() -> None:
    """同じシードなら何度実行しても同じ結果になること。"""
    cfg = TransactionsConfig(
        start_date=dt.date(2023, 1, 1), end_date=dt.date(2024, 6, 30), n_customers=20
    )
    first = generate(cfg, seed=3)
    second = generate(cfg, seed=3)
    assert first.transactions.equals(second.transactions)
    assert first.anomaly_labels.equals(second.anomaly_labels)


def test_different_seed_gives_different_data() -> None:
    """シードを変えれば結果が変わること（シードが効いていない事故の検出）。"""
    cfg = TransactionsConfig(
        start_date=dt.date(2023, 1, 1), end_date=dt.date(2024, 6, 30), n_customers=20
    )
    assert not generate(cfg, seed=3).transactions.equals(generate(cfg, seed=4).transactions)
