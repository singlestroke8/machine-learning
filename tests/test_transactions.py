"""取引明細（ローデータ）生成器のテスト。

生成器のテストで大事なのは「エラーなく動くこと」ではなく、
**業務ルールがデータに正しく反映されているか**である。
ここが崩れたデータで学習しても、モデルは業務と無関係なものを覚える。
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from demand_forecast.config import TransactionsConfig
from demand_forecast.data.transactions import (
    BRANCH,
    DEPARTMENTS,
    PRODUCTS,
    build_customers,
    build_sales_reps,
    business_days,
    generate_transactions,
    max_discount_for,
    summarize,
)
from demand_forecast.polars_utils import as_float

SMALL_CONFIG = TransactionsConfig(
    start_date=dt.date(2023, 1, 1),
    end_date=dt.date(2023, 12, 31),
    n_customers=10,
    reps_per_department=2,
)


@pytest.fixture(scope="module")
def transactions() -> pl.DataFrame:
    return generate_transactions(SMALL_CONFIG, seed=7)


# --- 再現性 ---------------------------------------------------------------


def test_generation_is_reproducible() -> None:
    """同じシードなら完全に同じ明細になること。"""
    assert generate_transactions(SMALL_CONFIG, seed=3).equals(
        generate_transactions(SMALL_CONFIG, seed=3)
    )


def test_different_seeds_give_different_data() -> None:
    assert not generate_transactions(SMALL_CONFIG, seed=1).equals(
        generate_transactions(SMALL_CONFIG, seed=2)
    )


def test_adding_customers_does_not_change_existing_ones() -> None:
    """顧客を増やしても、既存顧客の属性が変わらないこと。

    ここが崩れると、規模を変えるたびに過去の生成結果と比較できなくなる。
    """
    reps = build_sales_reps(SMALL_CONFIG.reps_per_department)
    small = build_customers(10, reps, seed=5)
    larger = build_customers(20, reps, seed=5)
    assert small == larger[:10]


# --- 業務ルール -----------------------------------------------------------


def test_sales_rep_owns_the_customer(transactions: pl.DataFrame) -> None:
    """1顧客につき担当営業は1名であること（担当が顧客を持つ）。"""
    per_customer = transactions.group_by("顧客コード").agg(
        pl.col("営業担当者").n_unique().alias("n_reps")
    )
    assert per_customer.get_column("n_reps").max() == 1


def test_rep_belongs_to_exactly_one_department(transactions: pl.DataFrame) -> None:
    """営業担当者は1つの部署にのみ属すること。"""
    per_rep = transactions.group_by("営業担当者").agg(
        pl.col("部署").n_unique().alias("n_departments")
    )
    assert per_rep.get_column("n_departments").max() == 1


def test_customers_have_repeat_transactions(transactions: pl.DataFrame) -> None:
    """同一顧客と繰り返し取引していること。"""
    order_days = transactions.group_by("顧客コード").agg(
        pl.col("受注日").n_unique().alias("n_days")
    )
    assert order_days.get_column("n_days").min() > 1


def test_departments_are_the_three_expected(transactions: pl.DataFrame) -> None:
    assert set(transactions.get_column("部署").unique().to_list()) == set(DEPARTMENTS)


def test_branch_is_tokyo_only(transactions: pl.DataFrame) -> None:
    """拠点は東京のみであること。"""
    assert transactions.get_column("拠点").unique().to_list() == [BRANCH]


def test_customers_are_in_kanto(transactions: pl.DataFrame) -> None:
    """顧客所在地が関東近県に収まっていること。"""
    kanto = {"東京都", "神奈川県", "埼玉県", "千葉県", "茨城県", "栃木県", "群馬県"}
    assert set(transactions.get_column("顧客所在地").unique().to_list()) <= kanto


def test_products_are_pc_or_software(transactions: pl.DataFrame) -> None:
    """商材が PC とソフトウェアに限られること。"""
    assert set(transactions.get_column("品名カテゴリ").unique().to_list()) == {
        "PC",
        "ソフトウェア",
    }
    assert set(transactions.get_column("品名").unique().to_list()) <= {p.name for p in PRODUCTS}


# --- 営業日 ---------------------------------------------------------------


def test_no_orders_on_weekends(transactions: pl.DataFrame) -> None:
    """土日に受注が立たないこと（法人取引なので営業日のみ）。"""
    weekdays = transactions.get_column("受注日").dt.weekday().unique().to_list()
    assert max(weekdays) <= 5, "土日に受注が発生しています"


def test_no_orders_on_holidays(transactions: pl.DataFrame) -> None:
    """祝日・年末年始に受注が立たないこと。"""
    valid = set(business_days(SMALL_CONFIG.start_date, SMALL_CONFIG.end_date))
    order_dates = set(transactions.get_column("受注日").unique().to_list())
    assert order_dates <= valid, f"営業日以外の受注: {sorted(order_dates - valid)[:5]}"


def test_business_days_exclude_new_year() -> None:
    days = set(business_days(dt.date(2023, 12, 25), dt.date(2024, 1, 10)))
    assert dt.date(2024, 1, 1) not in days
    assert dt.date(2024, 1, 4) in days  # 木曜、営業日


# --- 金額の整合性 ---------------------------------------------------------


def test_amounts_are_consistent(transactions: pl.DataFrame) -> None:
    """金額の計算が閉じていること。"""
    broken = transactions.filter(
        (pl.col("販売金額") != pl.col("販売単価") * pl.col("数量"))
        | (pl.col("原価") != pl.col("原価単価") * pl.col("数量"))
        | (pl.col("粗利") != pl.col("販売金額") - pl.col("原価"))
    )
    assert broken.is_empty(), f"金額が整合しない行が {broken.height} 件あります"


def test_discount_reduces_unit_price(transactions: pl.DataFrame) -> None:
    """値引率のぶんだけ販売単価が定価より安いこと。"""
    assert transactions.filter(pl.col("販売単価") > pl.col("定価")).is_empty()
    discounted = transactions.filter(pl.col("値引率") > 0.05)
    assert discounted.filter(pl.col("販売単価") >= pl.col("定価")).is_empty()


def test_no_negative_margin(transactions: pl.DataFrame) -> None:
    """原価割れ（逆ざや）の受注が発生しないこと。

    値引率が粗利率を超えると販売単価が原価単価を下回る。
    初版はこれを見落としており、PC の明細の3%が赤字になっていた。
    値引き上限を商材ごとに設けて解消したが、金額の整合性テストでは
    符号までは見ていなかったため気づけなかった。**符号は別に検査する。**
    """
    loss = transactions.filter(pl.col("粗利") < 0)
    assert loss.is_empty(), f"原価割れの明細が {loss.height} 件あります"
    assert transactions.filter(pl.col("販売単価") < pl.col("原価単価")).is_empty()


def test_discount_ceiling_differs_by_product(transactions: pl.DataFrame) -> None:
    """値引き上限が商材ごとに異なること（ハードは小さく、ソフトは大きい）。"""
    for product in PRODUCTS:
        ceiling = max_discount_for(product)
        actual = transactions.filter(pl.col("品名") == product.name).get_column("値引率")
        assert actual.max() <= ceiling + 1e-9, f"{product.name} が値引き上限を超えています"

    pc_ceiling = max(max_discount_for(p) for p in PRODUCTS if p.category == "PC")
    software_ceiling = min(max_discount_for(p) for p in PRODUCTS if p.category == "ソフトウェア")
    assert pc_ceiling < software_ceiling


def test_quantities_and_prices_are_positive(transactions: pl.DataFrame) -> None:
    assert transactions.get_column("数量").min() >= 1
    assert transactions.get_column("販売単価").min() > 0
    assert transactions.get_column("値引率").min() >= 0.0


def test_discount_rate_is_customer_specific(transactions: pl.DataFrame) -> None:
    """値引率が顧客ごとの取引条件になっていること。

    案件ごとの振れはあるが、顧客内のばらつきは顧客間のばらつきより小さいはず。
    """
    per_customer = transactions.group_by("顧客コード").agg(
        pl.col("値引率").mean().alias("mean"), pl.col("値引率").std().alias("std")
    )
    within = as_float(per_customer.get_column("std").mean())
    between = as_float(per_customer.get_column("mean").std())
    assert within < between, "顧客ごとの値引率の違いが表れていません"


# --- 需要構造 -------------------------------------------------------------


def test_department_product_affinity(transactions: pl.DataFrame) -> None:
    """部署ごとの得意商材が反映されていること。"""
    share = (
        transactions.group_by(["部署", "品名カテゴリ"])
        .len()
        .pivot(on="品名カテゴリ", index="部署", values="len")
        .with_columns((pl.col("PC") / (pl.col("PC") + pl.col("ソフトウェア"))).alias("pc_share"))
    )
    pc_share = dict(
        zip(
            share.get_column("部署").to_list(),
            share.get_column("pc_share").to_list(),
            strict=True,
        )
    )
    assert pc_share["営業1部"] > pc_share["営業2部"] > pc_share["ソリューション営業部"]


def test_fiscal_year_end_peak(transactions: pl.DataFrame) -> None:
    """年度末（3月）の受注が、期初（4月）より多いこと。"""
    by_month = (
        transactions.with_columns(pl.col("受注日").dt.month().alias("month"))
        .group_by("month")
        .agg(pl.col("販売金額").sum().alias("amount"))
    )
    amounts = dict(
        zip(
            by_month.get_column("month").to_list(),
            by_month.get_column("amount").to_list(),
            strict=True,
        )
    )
    assert amounts[3] > amounts[4]


def test_software_has_higher_margin_than_pc(transactions: pl.DataFrame) -> None:
    """ソフトウェアのほうが粗利率が高いこと（商材特性）。"""
    margin = transactions.group_by("品名カテゴリ").agg(
        (pl.col("粗利").sum() / pl.col("販売金額").sum()).alias("margin")
    )
    rates = dict(
        zip(
            margin.get_column("品名カテゴリ").to_list(),
            margin.get_column("margin").to_list(),
            strict=True,
        )
    )
    assert rates["ソフトウェア"] > rates["PC"]


# --- 検算とエラー処理 -----------------------------------------------------


def test_summary_reports_business_day_zero_rate(transactions: pl.DataFrame) -> None:
    """検算に、判断に使う営業日ベースの指標が含まれること。"""
    summary = summarize(transactions, SMALL_CONFIG)
    assert summary["明細行数"] == transactions.height
    assert summary["部署×品名の系列数"] == len(DEPARTMENTS) * len(PRODUCTS)
    assert "営業日のゼロ率" in summary


def test_too_many_reps_is_rejected() -> None:
    with pytest.raises(ValueError, match="営業担当者は最大"):
        build_sales_reps(reps_per_department=99)


def test_end_date_must_follow_start_date() -> None:
    with pytest.raises(ValueError, match="より後である必要があります"):
        TransactionsConfig(start_date=dt.date(2024, 1, 1), end_date=dt.date(2023, 1, 1))
