"""B2B 取引明細（ローデータ）の合成生成器。

想定する業務:

    東京の1拠点から、関東近県の法人顧客に PC とソフトウェアを販売する。
    営業担当が顧客を持ち、担当は3つの営業部署のいずれかに属する。
    値引率は顧客ごとの取引条件で決まり、同一顧客と繰り返し取引する。

出力は**集計前の取引明細**である。1行 = 1受注の1品目。
需要予測に使う形（日次の系列データ）への集計は、取り込み側の責務として分けている。
明細のまま持っておけば、系列の粒度（部署別・商品別・顧客別）を後から変えられるため。

法人取引の特徴として、次を明示的に組み込んでいる。

- **土日祝は受注ゼロ**（法人相手なので営業日にしか受注が立たない）
- 年度末（3月）・半期末（9月）の予算消化による増加
- 期初（4月・10月）と夏季休暇（8月）の落ち込み
- 月末（25日以降）に締めの受注が寄る
- 部署ごとの得意商材（営業1部=PC寄り、ソリューション営業部=ソフトウェア寄り）
- 顧客ごとの値引き交渉力と、値引きに応じた数量の増加（価格弾力性）
- 値引きは商材の粗利が確保できる範囲まで（原価割れの受注はしない）
- PC は薄利・ソフトウェアは高利益率という原価率の差

意図的に組み込んでいない要素（実データとの差分）:

- 失注・キャンセル・返品（数量が負になる行）
- 顧客の新規獲得と取引終了（期間中は顧客が固定）
- 納期遅延による計上月のずれ
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from demand_forecast.config import TransactionsConfig
from demand_forecast.features.calendar import japanese_holiday_flags

BRANCH = "東京本社"

DEPARTMENTS: tuple[str, ...] = ("営業1部", "営業2部", "ソリューション営業部")

PREFECTURES: tuple[str, ...] = (
    "東京都",
    "神奈川県",
    "埼玉県",
    "千葉県",
    "茨城県",
    "栃木県",
    "群馬県",
)

# 関東近県の顧客なので、東京・神奈川に偏らせる
PREFECTURE_WEIGHTS: tuple[float, ...] = (0.40, 0.20, 0.13, 0.12, 0.06, 0.05, 0.04)

_SURNAMES: tuple[str, ...] = (
    "佐藤",
    "鈴木",
    "高橋",
    "田中",
    "伊藤",
    "渡辺",
    "山本",
    "中村",
    "小林",
    "加藤",
    "吉田",
    "山田",
    "佐々木",
    "松本",
    "井上",
    "木村",
)


@dataclass(frozen=True)
class Product:
    """取扱商材。"""

    name: str
    category: str
    list_price: int
    cost_ratio: float
    #: 1受注あたりの基準数量。PC はまとめ買い、ソフトはライセンス数で幅がある
    base_quantity: float
    #: 年率の需要トレンド。PC は横ばい〜微減、ソフトは成長という想定
    yearly_growth: float


PRODUCTS: tuple[Product, ...] = (
    Product("ノートPC_標準モデル", "PC", 150_000, 0.84, 12.0, 0.02),
    Product("ノートPC_軽量モデル", "PC", 200_000, 0.82, 8.0, 0.10),
    Product("ノートPC_高性能モデル", "PC", 300_000, 0.80, 5.0, 0.06),
    Product("デスクトップPC_標準モデル", "PC", 120_000, 0.85, 9.0, -0.08),
    Product("ワークステーション", "PC", 500_000, 0.78, 2.5, 0.04),
    Product("オフィススイート", "ソフトウェア", 50_000, 0.55, 20.0, 0.05),
    Product("会計ソフト", "ソフトウェア", 180_000, 0.45, 6.0, 0.08),
    Product("CADソフト", "ソフトウェア", 400_000, 0.42, 3.0, 0.12),
    Product("セキュリティソフト", "ソフトウェア", 8_000, 0.50, 45.0, 0.15),
    Product("業務管理ソフト", "ソフトウェア", 250_000, 0.40, 4.0, 0.18),
)

# 部署ごとの商材の傾向。これがあることで「部署×商品」の系列に意味が出る
DEPARTMENT_AFFINITY: dict[str, dict[str, float]] = {
    "営業1部": {"PC": 1.7, "ソフトウェア": 0.4},
    "営業2部": {"PC": 1.0, "ソフトウェア": 1.0},
    "ソリューション営業部": {"PC": 0.35, "ソフトウェア": 1.9},
}

# 日本の法人取引の年間サイクル（1月〜12月）
MONTH_FACTOR: tuple[float, ...] = (
    0.85,  # 1月 年始で立ち上がりが遅い
    1.00,  # 2月
    1.80,  # 3月 年度末の予算消化
    0.70,  # 4月 期初で反動減
    0.90,  # 5月 連休
    1.05,  # 6月
    1.00,  # 7月
    0.75,  # 8月 夏季休暇
    1.35,  # 9月 半期末
    0.80,  # 10月 下期の立ち上がり
    1.00,  # 11月
    1.10,  # 12月 年末の駆け込み
)

# 月曜〜金曜。週明けは商談、週後半に受注が寄る
WEEKDAY_FACTOR: tuple[float, ...] = (0.90, 1.00, 1.05, 1.10, 1.15)

_MONTH_END_DAY = 25
_MONTH_START_DAY = 5
_MONTH_END_FACTOR = 1.35
_MONTH_START_FACTOR = 0.85

#: 値引きが数量を押し上げる強さ（価格弾力性の簡易表現）
_DISCOUNT_ELASTICITY = 1.8

#: 値引き後も確保する最低粗利率。これを下回る値引きはしない
_MIN_MARGIN_RATIO = 0.05

#: 商談上の値引き上限。原価に余裕があっても、これ以上は値引かない
_MAX_DISCOUNT = 0.35

#: 数量のばらつき（ガンマ・ポアソン混合の形状パラメータ）
_OVERDISPERSION_SHAPE = 6.0

# 乱数の名前空間。顧客属性と受注で別系列を使い、片方を変えても
# もう片方が動かないようにする（NumPy のシードは整数しか受け付けない）
_SEED_NS_CUSTOMER = 1
_SEED_NS_ORDER = 2

#: 顧客の受注頻度の基準。系列（部署×品名）あたりの日次明細数がこれで決まる。
#: 低すぎると営業日でもゼロが並び、日次の需要予測が成り立たなくなる。
_ORDER_RATE = 0.9


@dataclass(frozen=True)
class SalesRep:
    """営業担当者。"""

    name: str
    department: str


@dataclass(frozen=True)
class Customer:
    """顧客企業。

    担当営業が1名決まっており、そこから部署も一意に決まる
    （「担当営業が顧客を持つ」という業務ルールの表現）。
    """

    code: str
    name: str
    prefecture: str
    rep: SalesRep
    #: 値引き交渉力（0〜1）。商材ごとの値引き上限に対する比率として使う。
    #: 絶対値ではなく比率にすることで、商材ごとに上限が違っても
    #: 「A社は強気、B社は控えめ」という顧客差が保たれる。
    negotiation_power: float
    #: 受注の起きやすさ
    activity: float
    #: 1回あたりの購入規模
    size: float


def build_sales_reps(reps_per_department: int) -> tuple[SalesRep, ...]:
    """部署ごとに営業担当者を割り当てる。

    Raises:
        ValueError: 担当者数が用意した姓の数を超える場合。
    """
    total = reps_per_department * len(DEPARTMENTS)
    if total > len(_SURNAMES):
        msg = f"営業担当者は最大 {len(_SURNAMES)} 名までです（要求: {total} 名）"
        raise ValueError(msg)

    reps: list[SalesRep] = []
    for index, department in enumerate(DEPARTMENTS):
        for offset in range(reps_per_department):
            reps.append(SalesRep(_SURNAMES[index * reps_per_department + offset], department))
    return tuple(reps)


def build_customers(
    n_customers: int, reps: tuple[SalesRep, ...], seed: int
) -> tuple[Customer, ...]:
    """顧客マスタを作る。

    顧客ごとに独立した乱数系列を使うことで、顧客数を増やしても
    既存顧客の属性が変わらないようにしている。規模を変えたときに
    過去の生成結果と比較できなくなるのを避けるため。
    """
    customers: list[Customer] = []
    for index in range(n_customers):
        rng = np.random.default_rng((seed, _SEED_NS_CUSTOMER, index))
        prefecture = str(rng.choice(PREFECTURES, p=PREFECTURE_WEIGHTS))
        # 規模の大きい顧客ほど値引率が高く、受注も多いという相関を持たせる
        scale = float(rng.gamma(2.0, 0.5))
        customers.append(
            Customer(
                code=f"C{index + 1:03d}",
                name=f"{chr(ord('A') + index % 26)}{'' if index < 26 else index // 26}株式会社",
                prefecture=prefecture,
                rep=reps[index % len(reps)],
                negotiation_power=float(np.clip(0.15 + 0.35 * scale, 0.05, 1.0)),
                activity=float(np.clip(0.35 + 0.22 * scale, 0.1, 0.9)),
                size=float(np.clip(0.5 + 0.6 * scale, 0.3, 3.0)),
            )
        )
    return tuple(customers)


def business_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """営業日（平日かつ祝日でない日）を返す。"""
    days: list[dt.date] = []
    current = start
    while current <= end:
        days.append(current)
        current += dt.timedelta(days=1)

    holiday = japanese_holiday_flags(days)
    return [d for d, flag in zip(days, holiday, strict=True) if d.weekday() < 5 and flag == 0]


def max_discount_for(product: Product) -> float:
    """商材ごとの値引き上限を返す。

    原価率が高い商材ほど値引き余地が小さい。上限を超えると逆ざや（原価割れ）に
    なるため、最低粗利率を差し引いた値でクリップする。
    実務でも「ハードは値引き余地が小さく、ソフトは大きい」というのが通例で、
    この上限があることで PC とソフトウェアの値引き幅の差が自然に生まれる。
    """
    return min(1.0 - product.cost_ratio - _MIN_MARGIN_RATIO, _MAX_DISCOUNT)


def _seasonal_factor(day: dt.date) -> float:
    """月・曜日・月内位置から、その日の受注しやすさを求める。"""
    factor = MONTH_FACTOR[day.month - 1] * WEEKDAY_FACTOR[day.weekday()]
    if day.day >= _MONTH_END_DAY:
        factor *= _MONTH_END_FACTOR
    elif day.day <= _MONTH_START_DAY:
        factor *= _MONTH_START_FACTOR
    return factor


def _product_weights(department: str) -> np.ndarray:
    """部署の得意商材に応じた、商品の選ばれやすさを返す。"""
    weights = np.array(
        [DEPARTMENT_AFFINITY[department][p.category] * p.base_quantity**0.2 for p in PRODUCTS],
        dtype=np.float64,
    )
    return weights / weights.sum()


def generate_transactions(cfg: TransactionsConfig, seed: int = 42) -> pl.DataFrame:
    """取引明細を生成する。

    Args:
        cfg: 生成範囲（期間・顧客数・担当者数）。
        seed: 乱数シード。同じシードなら常に同じ明細になる。

    Returns:
        1行 = 1受注の1品目。受注日・顧客・担当・部署・品名・数量・金額を持つ。
        受注日順にソート済み。
    """
    reps = build_sales_reps(cfg.reps_per_department)
    customers = build_customers(cfg.n_customers, reps, seed)
    days = business_days(cfg.start_date, cfg.end_date)
    day_factors = np.array([_seasonal_factor(d) for d in days], dtype=np.float64)
    # トレンド計算の基準日からの経過年数
    elapsed_years = np.array([(d - cfg.start_date).days / 365.25 for d in days], dtype=np.float64)

    records: list[dict[str, object]] = []

    for customer_index, customer in enumerate(customers):
        rng = np.random.default_rng((seed, _SEED_NS_ORDER, customer_index))
        weights = _product_weights(customer.rep.department)

        # 受注が発生する日を、季節性で重み付けした確率で抽選する
        order_probability = np.clip(customer.activity * day_factors * _ORDER_RATE, 0.0, 0.95)
        has_order = rng.random(len(days)) < order_probability

        for day_index in np.flatnonzero(has_order):
            day = days[day_index]
            n_items = int(rng.integers(1, 4))
            chosen = rng.choice(len(PRODUCTS), size=n_items, replace=False, p=weights)

            for product_index in chosen:
                product = PRODUCTS[product_index]

                # 値引率は「商材ごとの上限 × 顧客の交渉力」を基準に、案件ごとに少し振れる。
                # 上限でクリップするので、原価割れ（逆ざや）の受注は発生しない。
                max_discount = max_discount_for(product)
                discount = float(
                    np.clip(
                        max_discount * customer.negotiation_power + rng.normal(0.0, 0.015),
                        0.0,
                        max_discount,
                    )
                )
                unit_price = int(round(product.list_price * (1.0 - discount), -1))
                cost_unit = int(round(product.list_price * product.cost_ratio, -1))

                trend = (1.0 + product.yearly_growth) ** elapsed_years[day_index]
                # 値引きが大きいほど数量が伸びる
                discount_effect = 1.0 + _DISCOUNT_ELASTICITY * discount
                lam = product.base_quantity * customer.size * trend * discount_effect
                noise = rng.gamma(_OVERDISPERSION_SHAPE, 1.0 / _OVERDISPERSION_SHAPE)
                quantity = int(rng.poisson(max(lam * noise, 0.2))) + 1

                amount = unit_price * quantity
                cost = cost_unit * quantity
                records.append(
                    {
                        "受注日": day,
                        "顧客コード": customer.code,
                        "顧客名": customer.name,
                        "顧客所在地": customer.prefecture,
                        "拠点": BRANCH,
                        "部署": customer.rep.department,
                        "営業担当者": customer.rep.name,
                        "品名": product.name,
                        "品名カテゴリ": product.category,
                        "数量": quantity,
                        "定価": product.list_price,
                        "値引率": round(discount, 4),
                        "販売単価": unit_price,
                        "販売金額": amount,
                        "原価単価": cost_unit,
                        "原価": cost,
                        "粗利": amount - cost,
                    }
                )

    if not records:
        msg = "取引明細が1件も生成されませんでした。期間または顧客数を見直してください。"
        raise ValueError(msg)

    return (
        pl.DataFrame(records)
        .with_columns(
            pl.col("受注日").cast(pl.Date),
            pl.col("数量").cast(pl.Int32),
            pl.col("定価").cast(pl.Int64),
            pl.col("値引率").cast(pl.Float64),
            pl.col("販売単価").cast(pl.Int64),
            pl.col("販売金額").cast(pl.Int64),
            pl.col("原価単価").cast(pl.Int64),
            pl.col("原価").cast(pl.Int64),
            pl.col("粗利").cast(pl.Int64),
        )
        .sort(["受注日", "顧客コード", "品名"])
    )


def summarize(transactions: pl.DataFrame, cfg: TransactionsConfig) -> dict[str, object]:
    """生成結果の検算値を返す。

    特に重要なのが「部署×品名で日次集計したときのゼロ日率」である。
    ここが高すぎると間欠需要の問題になり、日次の需要予測が成り立たない。
    データを作った直後にこれを確認できるようにしておく。
    """
    n_days = (cfg.end_date - cfg.start_date).days + 1
    n_business_days = len(business_days(cfg.start_date, cfg.end_date))
    series = transactions.select(["部署", "品名"]).unique().height

    daily = transactions.group_by(["部署", "品名", "受注日"]).agg(pl.col("数量").sum())
    observed = daily.height

    weekday_counts = (
        transactions.with_columns(pl.col("受注日").dt.weekday().alias("_dow"))
        .group_by("_dow")
        .len()
        .sort("_dow")
    )

    return {
        "明細行数": transactions.height,
        "期間": f"{cfg.start_date} 〜 {cfg.end_date}（{n_days}日）",
        "顧客数": transactions.get_column("顧客コード").n_unique(),
        "営業担当者数": transactions.get_column("営業担当者").n_unique(),
        "品名数": transactions.get_column("品名").n_unique(),
        "部署×品名の系列数": series,
        "営業日数": f"{n_business_days}（全 {n_days} 日中）",
        # 土日祝は法人取引が無いのが正しい姿なので、判断に使うのは営業日ベースの値
        "営業日の充足率": (
            f"{100 * observed / (series * n_business_days):.1f}%"
            f"（{observed:,}/{series * n_business_days:,}）"
        ),
        "営業日のゼロ率": f"{100 * (1 - observed / (series * n_business_days)):.1f}%",
        "全日ベースのゼロ率": (
            f"{100 * (1 - observed / (series * n_days)):.1f}%"
            "（土日祝が構造的にゼロなので、この値は高くて当然）"
        ),
        "曜日別受注件数": dict(
            zip(
                weekday_counts.get_column("_dow").to_list(),
                weekday_counts.get_column("len").to_list(),
                strict=True,
            )
        ),
        "総販売金額": int(transactions.get_column("販売金額").sum()),
        "平均粗利率": round(
            float(transactions.get_column("粗利").sum())
            / float(transactions.get_column("販売金額").sum()),
            4,
        ),
    }
