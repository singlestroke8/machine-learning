"""商品・組織・顧客のマスタ定義。

生成器の「登場人物」をここに集める。受注をどう発生させるかは
``generator`` 側の責務で、ここは静的な設定と顧客の生涯（取引開始・離反）だけを扱う。

顧客の生涯をここに置いているのは、**受注の生成とは独立に決まる**ためである。
「いつ取引が始まり、いつ細っていくか」は日々の受注の結果ではなく、
その顧客の事情で決まる。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

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
    "林",
    "斎藤",
    "清水",
    "山崎",
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
    #: 大型案件（全社更改）の対象になりうるか
    large_deal: bool


PRODUCTS: tuple[Product, ...] = (
    Product("ノートPC_標準モデル", "PC", 150_000, 0.84, 12.0, 0.02, True),
    Product("ノートPC_軽量モデル", "PC", 200_000, 0.82, 8.0, 0.10, True),
    Product("ノートPC_高性能モデル", "PC", 300_000, 0.80, 5.0, 0.06, False),
    Product("デスクトップPC_標準モデル", "PC", 120_000, 0.85, 9.0, -0.08, True),
    Product("ワークステーション", "PC", 500_000, 0.78, 2.5, 0.04, True),
    Product("オフィススイート", "ソフトウェア", 50_000, 0.55, 20.0, 0.05, True),
    Product("会計ソフト", "ソフトウェア", 180_000, 0.45, 6.0, 0.08, False),
    Product("CADソフト", "ソフトウェア", 400_000, 0.42, 3.0, 0.12, False),
    Product("セキュリティソフト", "ソフトウェア", 8_000, 0.50, 45.0, 0.15, True),
    Product("業務管理ソフト", "ソフトウェア", 250_000, 0.40, 4.0, 0.18, True),
)

PRODUCT_INDEX: dict[str, int] = {p.name: i for i, p in enumerate(PRODUCTS)}

#: 部署ごとの商材の傾向。これがあることで「部署×商品」に意味が出る
DEPARTMENT_AFFINITY: dict[str, dict[str, float]] = {
    "営業1部": {"PC": 1.7, "ソフトウェア": 0.4},
    "営業2部": {"PC": 1.0, "ソフトウェア": 1.0},
    "ソリューション営業部": {"PC": 0.35, "ソフトウェア": 1.9},
}

#: 関連購買。「この商材を買った顧客は、しばらく次の商材を買いやすい」
#: 導入に付帯して発生する購買を表す。クロスセルの課題（推薦）の土台になる。
FOLLOW_UP_AFFINITY: dict[str, dict[str, float]] = {
    "ノートPC_標準モデル": {"オフィススイート": 3.5, "セキュリティソフト": 3.0},
    "ノートPC_軽量モデル": {"オフィススイート": 3.5, "セキュリティソフト": 3.0},
    "ノートPC_高性能モデル": {"オフィススイート": 2.5, "CADソフト": 2.0},
    "デスクトップPC_標準モデル": {"オフィススイート": 3.0, "セキュリティソフト": 2.5},
    "ワークステーション": {"CADソフト": 4.0, "業務管理ソフト": 2.0},
    "CADソフト": {"ワークステーション": 2.5},
    "業務管理ソフト": {"会計ソフト": 2.5},
    "会計ソフト": {"業務管理ソフト": 2.0},
}

#: 関連購買が効く期間（暦日）
FOLLOW_UP_WINDOW_DAYS = 90

#: 日本の法人取引の年間サイクル（1月〜12月）。年ごとに揺らぎを掛ける
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

#: 月曜〜金曜。週明けは商談、週後半に受注が寄る
WEEKDAY_FACTOR: tuple[float, ...] = (0.90, 1.00, 1.05, 1.10, 1.15)

MONTH_END_DAY = 25
MONTH_START_DAY = 5
MONTH_END_FACTOR = 1.35
MONTH_START_FACTOR = 0.85

#: 値引きが数量を押し上げる強さ（価格弾力性の簡易表現）
DISCOUNT_ELASTICITY = 1.8

#: 値引き後も確保する最低粗利率。通常の受注はこれを下回らない
MIN_MARGIN_RATIO = 0.05

#: 商談上の値引き上限。原価に余裕があっても、これ以上は値引かない
MAX_DISCOUNT = 0.35

#: 数量のばらつき（ガンマ・ポアソン混合の形状パラメータ）。小さいほど裾が重い
OVERDISPERSION_SHAPE = 4.0

#: 顧客あたりの受注頻度の基準
ORDER_RATE = 0.30

# 乱数の名前空間。用途ごとに別系列を使い、片方を変えても他が動かないようにする
SEED_NS_YEAR = 0
SEED_NS_CUSTOMER = 1
SEED_NS_ORDER = 2
SEED_NS_LARGE = 3
SEED_NS_ANOMALY = 4


def max_discount_for(product: Product) -> float:
    """商材ごとの値引き上限を返す。

    原価率が高い商材ほど値引き余地が小さい。上限を超えると原価割れになる。
    実務でも「ハードは値引き余地が小さく、ソフトは大きい」というのが通例で、
    この上限があることで PC とソフトウェアの値引き幅の差が自然に生まれる。
    """
    return min(1.0 - product.cost_ratio - MIN_MARGIN_RATIO, MAX_DISCOUNT)


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

    ``entry_month`` と ``churn_start`` が顧客の生涯を決める。
    **どちらもデータには直接現れない**（受注の有無としてしか観測できない）。
    離反予測の課題では、これを受注履歴から推し当てることになる。
    """

    code: str
    name: str
    prefecture: str
    rep: SalesRep
    #: 値引き交渉力（0〜1）。商材ごとの値引き上限に対する比率として使う
    negotiation_power: float
    #: 受注の起きやすさ
    activity: float
    #: 1受注あたりの規模
    size: float
    #: 取引開始月。期首より後なら「期間の途中で獲得した新規顧客」
    entry_month: dt.date
    #: 先細りが始まる月。``None`` なら最後まで取引が続く
    churn_start: dt.date | None
    #: 先細りにかける月数
    churn_months: int
    #: 担当者が交代する月。``None`` なら交代しない
    rep_change_month: dt.date | None
    #: 交代後の担当者
    next_rep: SalesRep | None


def build_sales_reps(reps_per_department: int) -> tuple[SalesRep, ...]:
    """営業担当者を作る。部署ごとに同数を割り当てる。"""
    needed = reps_per_department * len(DEPARTMENTS)
    if needed > len(_SURNAMES):
        msg = f"営業担当者が多すぎます（最大 {len(_SURNAMES)} 名）: {needed} 名"
        raise ValueError(msg)
    reps: list[SalesRep] = []
    for index, department in enumerate(DEPARTMENTS):
        for offset in range(reps_per_department):
            reps.append(SalesRep(_SURNAMES[index * reps_per_department + offset], department))
    return tuple(reps)


def _month_floor(day: dt.date) -> dt.date:
    return day.replace(day=1)


def _add_months(day: dt.date, months: int) -> dt.date:
    total = (day.year * 12 + day.month - 1) + months
    return dt.date(total // 12, total % 12 + 1, 1)


def build_customers(
    *,
    n_customers: int,
    reps: tuple[SalesRep, ...],
    start: dt.date,
    end: dt.date,
    new_customer_ratio: float,
    churn_ratio: float,
    seed: int,
) -> tuple[Customer, ...]:
    """顧客マスタを作る。

    顧客ごとに独立した乱数系列を使うことで、顧客数を増やしても
    既存顧客の属性が変わらないようにしている。規模を変えたときに
    過去の生成結果と比較できなくなるのを避けるため。

    離反は**単一の原因で起きないようにしている**。担当者が交代した顧客は
    離反しやすいが、交代しても離れない顧客も、交代せずに離れる顧客もいる。
    1つの列を見れば分かるデータにすると、機械学習の題材にならない。
    """
    n_months = (end.year * 12 + end.month) - (start.year * 12 + start.month)
    first_month = _month_floor(start)
    customers: list[Customer] = []

    for index in range(n_customers):
        rng = np.random.default_rng((seed, SEED_NS_CUSTOMER, index))
        prefecture = str(rng.choice(PREFECTURES, p=PREFECTURE_WEIGHTS))
        # 規模の大きい顧客ほど値引率が高く、受注も多いという相関を持たせる
        scale = float(rng.gamma(2.0, 0.5))

        # --- 担当者の交代（離反の原因の1つだが、決定的ではない） ---
        rep = reps[index % len(reps)]
        rep_change_month: dt.date | None = None
        next_rep: SalesRep | None = None
        if rng.random() < 0.15:
            offset = int(rng.integers(max(1, n_months // 4), max(2, n_months - 3)))
            rep_change_month = _add_months(first_month, offset)
            same_dept = [r for r in reps if r.department == rep.department and r != rep]
            if same_dept:
                next_rep = same_dept[int(rng.integers(0, len(same_dept)))]
            else:
                rep_change_month = None

        # --- 取引開始（新規獲得） ---
        entry_month = first_month
        is_new = rng.random() < new_customer_ratio
        if is_new and n_months > 12:
            offset = int(rng.integers(2, max(3, int(n_months * 0.8))))
            entry_month = _add_months(first_month, offset)

        # --- 離反（先細り） ---
        churn_start: dt.date | None = None
        churn_months = 0
        churn_probability = churn_ratio * (1.8 if rep_change_month is not None else 1.0)
        # 新規顧客も離反しうるが、入ってすぐ細るのは不自然なので余裕を見る
        earliest = (
            (entry_month.year * 12 + entry_month.month)
            - (first_month.year * 12 + first_month.month)
            + 6
        )
        if rng.random() < churn_probability and earliest < n_months - 2:
            offset = int(rng.integers(earliest, n_months - 1))
            churn_start = _add_months(first_month, offset)
            churn_months = int(rng.integers(3, 7))

        customers.append(
            Customer(
                code=f"C{index + 1:04d}",
                name=f"{chr(ord('A') + index % 26)}{'' if index < 26 else index // 26}株式会社",
                prefecture=prefecture,
                rep=rep,
                negotiation_power=float(np.clip(0.15 + 0.35 * scale, 0.05, 1.0)),
                activity=float(np.clip(0.35 + 0.22 * scale, 0.1, 0.9)),
                size=float(np.clip(0.5 + 0.6 * scale, 0.3, 3.0)),
                entry_month=entry_month,
                churn_start=churn_start,
                churn_months=churn_months,
                rep_change_month=rep_change_month,
                next_rep=next_rep,
            )
        )
    return tuple(customers)


def lifecycle_factors(customer: Customer, day: dt.date) -> tuple[float, float, float]:
    """その日の顧客の状態を ``(受注頻度, 規模, 値引き圧力)`` の倍率で返す。

    先細り中の顧客には3つの予兆が同時に現れる。

    1. 受注の間隔が伸びる（頻度が落ちる）
    2. 1回あたりが小さくなる（小口化）
    3. 値引き要求が強まる（他社と比較され始める）

    **どれか1つを見れば分かる、という形にしていない。**
    実務でも離反の予兆は複数の弱い信号として現れる。
    """
    if day < customer.entry_month:
        return 0.0, 1.0, 0.0
    if customer.churn_start is None or day < customer.churn_start:
        return 1.0, 1.0, 0.0

    elapsed = (day.year * 12 + day.month) - (
        customer.churn_start.year * 12 + customer.churn_start.month
    )
    progress = min(1.0, elapsed / customer.churn_months)
    return (
        max(0.0, 1.0 - progress),
        1.0 - 0.5 * progress,
        0.25 * progress,
    )


def rep_on(customer: Customer, day: dt.date) -> SalesRep:
    """その日の担当営業を返す。交代月以降は後任になる。"""
    change, successor = customer.rep_change_month, customer.next_rep
    if change is not None and successor is not None and day >= change:
        return successor
    return customer.rep
