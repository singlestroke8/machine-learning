"""取引明細（ローデータ）の生成。

実務で受け取るのと同じ「1行 = 1受注の1品目」の形を作る。
集計済みのきれいなデータではなく、**集計する前の姿**から始めるための土台。

設計の原則は1つだけである。

> **構造は仕込むが、1つの列を見れば分かる形にはしない。**

季節性や成長率を定数で書き込むだけだと、分析しても定数がそのまま出てくる。
それでは機械学習の題材にならない。ここでは要因を複数絡ませ、
年ごとの揺らぎと大型案件で、素朴な予測が簡単には当たらないようにしている。

出力は3ファイル。

============================ ==========================================
``transactions.csv``         取引明細。学習にも分析にも使う
``customers.csv``            顧客マスタ。業務的に既知の属性だけを持つ
``anomaly_labels.csv``       異常の正解。**評価にのみ使い、学習には使わない**
============================ ==========================================

異常ラベルをファイルごと分けているのは、うっかり特徴量に混ぜないための
構造的な防止策である。実務では正解ラベルなど存在しないので、
同じ扱いができるようにしておく。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from sales_analytics.config import TransactionsConfig
from sales_analytics.data.masters import (
    BRANCH,
    DEPARTMENT_AFFINITY,
    DISCOUNT_ELASTICITY,
    FOLLOW_UP_AFFINITY,
    FOLLOW_UP_WINDOW_DAYS,
    MONTH_END_DAY,
    MONTH_END_FACTOR,
    MONTH_FACTOR,
    MONTH_START_DAY,
    MONTH_START_FACTOR,
    ORDER_RATE,
    OVERDISPERSION_SHAPE,
    PRODUCT_INDEX,
    PRODUCTS,
    SEED_NS_ANOMALY,
    SEED_NS_LARGE,
    SEED_NS_ORDER,
    SEED_NS_YEAR,
    WEEKDAY_FACTOR,
    Customer,
    build_customers,
    build_sales_reps,
    lifecycle_factors,
    max_discount_for,
    rep_on,
)
from sales_analytics.features.calendar import business_days

# --- 年ごとの揺らぎ ---------------------------------------------------------
#: 年ごとの水準変動（対数正規のばらつき）。景況や競合状況の変化を表す
YEAR_LEVEL_SIGMA = 0.09
#: 月係数の年ごとの揺らぎ。「毎年3月が同じ倍率」にしないための項
MONTH_JITTER_SIGMA = 0.13

# --- 大型案件 ---------------------------------------------------------------
#: 1ヶ月あたりの大型案件の期待件数（全社）
LARGE_DEAL_PER_MONTH = 7.0
#: 大型案件の数量倍率の範囲
LARGE_DEAL_MULTIPLIER = (25.0, 90.0)

# --- 異常・返品 -------------------------------------------------------------
#: 数量の桁違い（入力ミス）の割合
TYPO_RATE = 0.003
#: 値引率の異常（承認漏れ）の割合。原価割れになるので粗利で検出できる
DISCOUNT_ERROR_RATE = 0.002
#: 期末の押し込み受注（翌月に取消される）の割合
PUSH_SALE_RATE = 0.002
#: 通常の返品の割合。**異常ではない**。異常検知を難しくするために入れる
RETURN_RATE = 0.005

ANOMALY_KINDS: tuple[str, ...] = ("quantity_typo", "discount_error", "push_sale")


@dataclass
class Line:
    """明細1行。

    **金額を持たない**のが要点である。販売金額・原価・粗利は数量と単価から
    最後にまとめて導出する。金額を別々に持つと、数量を書き換えたときに
    更新し忘れて「計算は合っているが業務的にありえない行」が生まれる。
    実際、前身のコードではそれで原価割れの受注が3%混入した。
    """

    order_id: str
    line_no: int
    day: dt.date
    customer_code: str
    customer_name: str
    prefecture: str
    department: str
    rep_name: str
    product: str
    category: str
    quantity: int
    list_price: int
    discount: float
    unit_price: int
    cost_unit: int


@dataclass(frozen=True)
class GeneratedData:
    """生成結果一式。"""

    transactions: pl.DataFrame
    customers: pl.DataFrame
    anomaly_labels: pl.DataFrame


def _year_adjustments(
    years: list[int], seed: int
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """年ごとの水準と、月係数の年ごとの揺らぎを作る。

    これが無いと「前年同月と同じ」が当たりすぎ、機械学習の出番が無くなる。
    実務の販売データで前年同月比が15〜25%外すのは、この揺らぎがあるため。
    """
    rng = np.random.default_rng((seed, SEED_NS_YEAR))
    level = 1.0
    year_level: dict[int, float] = {}
    month_jitter: dict[tuple[int, int], float] = {}
    for year in years:
        level *= float(np.exp(rng.normal(0.0, YEAR_LEVEL_SIGMA)))
        year_level[year] = level
        for month in range(1, 13):
            month_jitter[(year, month)] = float(np.exp(rng.normal(0.0, MONTH_JITTER_SIGMA)))
    return year_level, month_jitter


def _day_factor(
    day: dt.date,
    year_level: dict[int, float],
    month_jitter: dict[tuple[int, int], float],
) -> float:
    """その営業日の受注の起きやすさ。季節性・曜日・月内の位置を掛け合わせる。"""
    factor = MONTH_FACTOR[day.month - 1] * month_jitter[(day.year, day.month)]
    factor *= year_level[day.year]
    factor *= WEEKDAY_FACTOR[day.weekday()]
    if day.day >= MONTH_END_DAY:
        factor *= MONTH_END_FACTOR
    elif day.day <= MONTH_START_DAY:
        factor *= MONTH_START_FACTOR
    return float(factor)


def _base_weights(department: str) -> np.ndarray:
    """部署ごとの商材の選ばれやすさ。"""
    weights = np.array(
        [DEPARTMENT_AFFINITY[department][p.category] * p.base_quantity**0.2 for p in PRODUCTS],
        dtype=np.float64,
    )
    return weights / weights.sum()


def _apply_follow_up(
    weights: np.ndarray, recent: list[tuple[dt.date, str]], day: dt.date
) -> np.ndarray:
    """直近に買った商材に応じて、次に買いやすい商材の重みを上げる。

    PC を入れた顧客は、しばらくして Office やセキュリティを買う。
    この依存があることで「次に何を提案すべきか」の課題が成立する。
    """
    if not recent:
        return weights
    adjusted = weights.copy()
    for bought_at, name in recent:
        if (day - bought_at).days > FOLLOW_UP_WINDOW_DAYS:
            continue
        for target, boost in FOLLOW_UP_AFFINITY.get(name, {}).items():
            adjusted[PRODUCT_INDEX[target]] *= boost
    total = float(adjusted.sum())
    return adjusted / total if total > 0 else weights


def _generate_orders(
    customers: tuple[Customer, ...],
    days: list[dt.date],
    year_level: dict[int, float],
    month_jitter: dict[tuple[int, int], float],
    seed: int,
) -> list[Line]:
    """顧客ごとに、日々の受注を生成する。"""
    day_factors = np.array([_day_factor(d, year_level, month_jitter) for d in days])
    elapsed_years = np.array([(d - days[0]).days / 365.25 for d in days])
    lines: list[Line] = []

    for customer_index, customer in enumerate(customers):
        rng = np.random.default_rng((seed, SEED_NS_ORDER, customer_index))
        recent: list[tuple[dt.date, str]] = []
        draws = rng.random(len(days))

        for day_index, day in enumerate(days):
            frequency, size_scale, discount_pressure = lifecycle_factors(customer, day)
            if frequency <= 0.0:
                continue
            probability = min(
                0.95, customer.activity * day_factors[day_index] * ORDER_RATE * frequency
            )
            if draws[day_index] >= probability:
                continue

            rep = rep_on(customer, day)
            weights = _apply_follow_up(_base_weights(rep.department), recent, day)
            n_items = int(rng.integers(1, 4))
            chosen = rng.choice(len(PRODUCTS), size=n_items, replace=False, p=weights)
            order_id = f"SO{day:%Y%m%d}-{customer.code}"

            for line_no, product_index in enumerate(chosen, start=1):
                product = PRODUCTS[int(product_index)]
                ceiling = max_discount_for(product)
                power = min(1.0, customer.negotiation_power + discount_pressure)
                discount = float(np.clip(ceiling * power + rng.normal(0.0, 0.015), 0.0, ceiling))
                trend = (1.0 + product.yearly_growth) ** elapsed_years[day_index]
                lam = (
                    product.base_quantity
                    * customer.size
                    * size_scale
                    * trend
                    * (1.0 + DISCOUNT_ELASTICITY * discount)
                )
                noise = float(rng.gamma(OVERDISPERSION_SHAPE, 1.0 / OVERDISPERSION_SHAPE))
                lines.append(
                    Line(
                        order_id=order_id,
                        line_no=line_no,
                        day=day,
                        customer_code=customer.code,
                        customer_name=customer.name,
                        prefecture=customer.prefecture,
                        department=rep.department,
                        rep_name=rep.name,
                        product=product.name,
                        category=product.category,
                        quantity=int(rng.poisson(max(lam * noise, 0.2))) + 1,
                        list_price=product.list_price,
                        discount=discount,
                        unit_price=int(round(product.list_price * (1.0 - discount), -1)),
                        cost_unit=int(round(product.list_price * product.cost_ratio, -1)),
                    )
                )
                recent.append((day, product.name))
            recent = [(d, n) for d, n in recent if (day - d).days <= FOLLOW_UP_WINDOW_DAYS]

    return lines


def _inject_large_deals(
    lines: list[Line],
    customers: tuple[Customer, ...],
    days: list[dt.date],
    seed: int,
) -> None:
    """大型案件（全社更改）を注入する。

    月次の着地を大きく振らせる要因であり、同時に**異常検知にとって
    紛らわしい「珍しいが正常な」データ**でもある。
    異常検知が難しいのは異常を見つけることではなく、これを誤検知しないこと。
    """
    rng = np.random.default_rng((seed, SEED_NS_LARGE))
    n_months = len({(d.year, d.month) for d in days})
    n_deals = int(rng.poisson(LARGE_DEAL_PER_MONTH * n_months))
    eligible = [
        i for i, line in enumerate(lines) if PRODUCTS[PRODUCT_INDEX[line.product]].large_deal
    ]
    if not eligible or n_deals <= 0:
        return

    # 規模の大きい顧客ほど全社更改が起きやすい
    size_by_code = {c.code: c.size for c in customers}
    weights = np.array([size_by_code[lines[i].customer_code] for i in eligible])
    weights = weights / weights.sum()
    picked = rng.choice(len(eligible), size=min(n_deals, len(eligible)), replace=False, p=weights)

    for slot in picked:
        line = lines[eligible[int(slot)]]
        multiplier = float(rng.uniform(*LARGE_DEAL_MULTIPLIER))
        line.quantity = max(1, round(line.quantity * multiplier))


def _negative_copy(line: Line, day: dt.date, prefix: str) -> Line:
    """取消・返品の行を作る。数量を負にした同じ内容の行。"""
    return Line(
        order_id=f"{prefix}{day:%Y%m%d}-{line.customer_code}",
        line_no=line.line_no,
        day=day,
        customer_code=line.customer_code,
        customer_name=line.customer_name,
        prefecture=line.prefecture,
        department=line.department,
        rep_name=line.rep_name,
        product=line.product,
        category=line.category,
        quantity=-line.quantity,
        list_price=line.list_price,
        discount=line.discount,
        unit_price=line.unit_price,
        cost_unit=line.cost_unit,
    )


def _inject_anomalies(
    lines: list[Line], days: list[dt.date], seed: int
) -> tuple[list[Line], list[dict[str, object]]]:
    """異常と返品を注入し、異常の正解ラベルを返す。

    3種類の異常を入れる。

    ``quantity_typo``
        数量の桁違い。入力ミス。金額だけ見ると大型案件と見分けがつかない。
    ``discount_error``
        値引率が承認上限を超えている。**原価割れになるので粗利で検出できる**。
    ``push_sale``
        期末の押し込み受注。翌月に取消される。1行だけ見ても分からない。

    加えて、**異常ではない通常の返品**も入れる。
    「数量が負」だけを異常とみなす素朴な検知が誤検知するようにするため。
    """
    rng = np.random.default_rng((seed, SEED_NS_ANOMALY))
    n = len(lines)
    labels: list[dict[str, object]] = []
    extra: list[Line] = []

    def _label(line: Line, kind: str) -> None:
        labels.append({"受注番号": line.order_id, "明細番号": line.line_no, "異常種別": kind})

    used: set[int] = set()

    def _pick(rate: float) -> list[int]:
        count = max(1, int(n * rate))
        chosen: list[int] = []
        while len(chosen) < count:
            index = int(rng.integers(0, n))
            if index not in used:
                used.add(index)
                chosen.append(index)
        return chosen

    # 1. 数量の桁違い
    for index in _pick(TYPO_RATE):
        lines[index].quantity *= 10
        _label(lines[index], "quantity_typo")

    # 2. 値引率の異常（承認漏れ）→ 原価割れになる
    for index in _pick(DISCOUNT_ERROR_RATE):
        line = lines[index]
        product = PRODUCTS[PRODUCT_INDEX[line.product]]
        line.discount = round(
            min(0.9, max_discount_for(product) + float(rng.uniform(0.10, 0.25))), 4
        )
        line.unit_price = int(round(line.list_price * (1.0 - line.discount), -1))
        _label(line, "discount_error")

    # 3. 期末の押し込み → 翌月に取消
    month_end = [i for i in range(n) if lines[i].day.day >= MONTH_END_DAY and i not in used]
    if month_end:
        count = max(1, int(n * PUSH_SALE_RATE))
        for slot in rng.choice(len(month_end), size=min(count, len(month_end)), replace=False):
            line = lines[month_end[int(slot)]]
            used.add(month_end[int(slot)])
            later = [d for d in days if line.day < d <= line.day + dt.timedelta(days=35)]
            if not later:
                continue
            cancel_day = later[int(rng.integers(0, len(later)))]
            cancel = _negative_copy(line, cancel_day, "CN")
            extra.append(cancel)
            _label(line, "push_sale")
            _label(cancel, "push_sale")

    # 4. 通常の返品（異常ではない）
    count = max(1, int(n * RETURN_RATE))
    for slot in rng.choice(n, size=count, replace=False):
        line = lines[int(slot)]
        later = [d for d in days if line.day < d <= line.day + dt.timedelta(days=60)]
        if not later:
            continue
        return_day = later[int(rng.integers(0, len(later)))]
        returned = _negative_copy(line, return_day, "RT")
        # 全量返品ではなく一部返品のことが多い
        returned.quantity = -max(1, abs(returned.quantity) // int(rng.integers(2, 5)))
        extra.append(returned)

    return extra, labels


def _to_frame(lines: list[Line]) -> pl.DataFrame:
    """明細を DataFrame にする。金額はここで**まとめて導出する**。"""
    frame = pl.DataFrame(
        [
            {
                "受注番号": line.order_id,
                "明細番号": line.line_no,
                "受注日": line.day,
                "顧客コード": line.customer_code,
                "顧客名": line.customer_name,
                "顧客所在地": line.prefecture,
                "拠点": BRANCH,
                "部署": line.department,
                "営業担当者": line.rep_name,
                "品名": line.product,
                "品名カテゴリ": line.category,
                "数量": line.quantity,
                "定価": line.list_price,
                "値引率": round(line.discount, 4),
                "販売単価": line.unit_price,
                "原価単価": line.cost_unit,
            }
            for line in lines
        ]
    )
    return (
        frame.with_columns(
            (pl.col("販売単価") * pl.col("数量")).alias("販売金額"),
            (pl.col("原価単価") * pl.col("数量")).alias("原価"),
        )
        .with_columns((pl.col("販売金額") - pl.col("原価")).alias("粗利"))
        .select(
            "受注番号",
            "明細番号",
            "受注日",
            "顧客コード",
            "顧客名",
            "顧客所在地",
            "拠点",
            "部署",
            "営業担当者",
            "品名",
            "品名カテゴリ",
            "数量",
            "定価",
            "値引率",
            "販売単価",
            "販売金額",
            "原価単価",
            "原価",
            "粗利",
        )
        .sort("受注日", "受注番号", "明細番号")
    )


def _customers_frame(customers: tuple[Customer, ...]) -> pl.DataFrame:
    """顧客マスタ。**業務的に既知の属性だけ**を書き出す。

    離反するかどうか（``churn_start``）は書かない。それは結果であって
    マスタに載っている情報ではない。ここに書くとリークになる。
    """
    return pl.DataFrame(
        [
            {
                "顧客コード": c.code,
                "顧客名": c.name,
                "顧客所在地": c.prefecture,
                "拠点": BRANCH,
                "取引開始月": c.entry_month,
            }
            for c in customers
        ]
    ).sort("顧客コード")


def generate(cfg: TransactionsConfig, seed: int = 42) -> GeneratedData:
    """取引明細・顧客マスタ・異常ラベルを生成する。

    Args:
        cfg: 業務設定（期間・顧客数・新規/離反の割合）。
        seed: 乱数シード。同じ値なら何度実行しても同じ結果になる。
    """
    days = business_days(cfg.start_date, cfg.end_date)
    if not days:
        msg = f"営業日が1日もありません: {cfg.start_date} 〜 {cfg.end_date}"
        raise ValueError(msg)

    reps = build_sales_reps(cfg.reps_per_department)
    customers = build_customers(
        n_customers=cfg.n_customers,
        reps=reps,
        start=cfg.start_date,
        end=cfg.end_date,
        new_customer_ratio=cfg.new_customer_ratio,
        churn_ratio=cfg.churn_ratio,
        seed=seed,
    )
    years = sorted({d.year for d in days})
    year_level, month_jitter = _year_adjustments(years, seed)

    lines = _generate_orders(customers, days, year_level, month_jitter, seed)
    if not lines:
        msg = "受注が1件も生成されませんでした。設定を見直してください。"
        raise ValueError(msg)

    _inject_large_deals(lines, customers, days, seed)
    extra, labels = _inject_anomalies(lines, days, seed)
    lines.extend(extra)

    return GeneratedData(
        transactions=_to_frame(lines),
        customers=_customers_frame(customers),
        anomaly_labels=pl.DataFrame(
            labels, schema={"受注番号": pl.Utf8, "明細番号": pl.Int64, "異常種別": pl.Utf8}
        ).sort("受注番号", "明細番号"),
    )
