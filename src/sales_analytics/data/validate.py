"""生成したデータが、課題を成立させる条件を満たしているかを検査する。

**合格基準を先に決めてから作る**ためのモジュール。
前身のコードでは、生成したデータに原価割れの受注が3%混ざっていたことに
しばらく気づかなかった。「動いた」ことと「使えるデータができた」ことは別である。

ここで最も重要なのは ``前年同月比の外し率`` である。
これが低すぎるデータは、規則的すぎて機械学習の題材にならない。
Excel の前年同月比に負けるモデルを作っても意味がない。
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from sales_analytics.polars_utils import as_date, as_float

#: 「前年同月 × 成長率」で予測したときの外し率の許容範囲。
#: 低すぎる = 規則的すぎて機械学習が勝てない。高すぎる = 何をやっても当たらない。
YOY_WAPE_RANGE = (0.15, 0.25)
#: 離反顧客の割合の許容範囲。低すぎると分類の学習が成立しない
CHURN_RATIO_RANGE = (0.10, 0.25)
#: 異常の混入率。少なすぎると評価できず、多すぎると「異常」ではなくなる
ANOMALY_RATIO_RANGE = (0.003, 0.02)
#: 顧客あたり・月あたりの受注回数の下限。
#: 離反予測は「受注が途絶えたこと」を見るので、平常時に十分な頻度が要る。
#: 頻度が低いと「たまたま今月無かった」と「離反」が区別できない。
MIN_ORDERS_PER_CUSTOMER_MONTH = 1.0
#: 大型案件（通常の10倍以上の金額）の割合
LARGE_DEAL_RATIO_RANGE = (0.005, 0.03)

#: 離反の判定に使う「直近この日数、受注が無ければ離反」
CHURN_WINDOW_DAYS = 120


@dataclass(frozen=True)
class Check:
    """検査項目1つ分の結果。"""

    name: str
    value: str
    passed: bool
    note: str


def _yoy_wape(transactions: pl.DataFrame) -> tuple[float, int]:
    """全社の月次売上を「前年同月 × 全体の成長率」で予測したときの外し率。"""
    monthly = (
        transactions.group_by(pl.col("受注日").dt.truncate("1mo").alias("月"))
        .agg(pl.col("販売金額").sum().alias("売上"))
        .sort("月")
    )
    joined = monthly.with_columns(pl.col("売上").shift(12).alias("前年")).drop_nulls("前年")
    if joined.height == 0:
        return float("nan"), 0
    growth = as_float(joined.get_column("売上").sum()) / as_float(joined.get_column("前年").sum())
    predicted = joined.get_column("前年") * growth
    actual = joined.get_column("売上")
    error = as_float((actual - predicted).abs().sum())
    return error / as_float(actual.sum()), joined.height


def _churn_ratio(transactions: pl.DataFrame) -> float:
    """期末時点で「直近 CHURN_WINDOW_DAYS 受注が無い」顧客の割合。"""
    end = as_date(transactions.get_column("受注日").max())
    last = transactions.group_by("顧客コード").agg(pl.col("受注日").max().alias("最終受注"))
    churned = last.filter(
        pl.col("最終受注") < pl.lit(end) - pl.duration(days=CHURN_WINDOW_DAYS)
    ).height
    return churned / last.height


def _empty_department_months(transactions: pl.DataFrame) -> int:
    """部署×月で受注が1件も無かった組み合わせの数。

    月次の着地予測は「部署×月」が予測の単位なので、ここに穴があると成立しない。
    """
    months = transactions.get_column("受注日").dt.truncate("1mo").n_unique()
    departments = transactions.get_column("部署").n_unique()
    filled = (
        transactions.filter(pl.col("数量") > 0)
        .group_by("部署", pl.col("受注日").dt.truncate("1mo"))
        .agg(pl.len())
        .height
    )
    return months * departments - filled


def _orders_per_customer_month(transactions: pl.DataFrame) -> float:
    """顧客1社あたり、1ヶ月あたりの受注回数（明細ではなく受注の数）。"""
    orders = transactions.filter(pl.col("数量") > 0).get_column("受注番号").n_unique()
    customers = transactions.get_column("顧客コード").n_unique()
    months = transactions.get_column("受注日").dt.truncate("1mo").n_unique()
    return orders / (customers * months)


def _large_deal_ratio(transactions: pl.DataFrame) -> float:
    """金額が全体の中央値の10倍を超える明細の割合。"""
    positive = transactions.filter(pl.col("販売金額") > 0)
    threshold = as_float(positive.get_column("販売金額").median()) * 10
    return positive.filter(pl.col("販売金額") > threshold).height / positive.height


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def run_checks(transactions: pl.DataFrame, anomaly_labels: pl.DataFrame) -> list[Check]:
    """すべての合格基準を検査する。"""
    yoy, n_months = _yoy_wape(transactions)
    churn = _churn_ratio(transactions)
    anomaly_rows = anomaly_labels.select("受注番号", "明細番号").unique().height
    anomaly_ratio = anomaly_rows / transactions.height
    empty_months = _empty_department_months(transactions)
    frequency = _orders_per_customer_month(transactions)
    large = _large_deal_ratio(transactions)

    # 異常ラベルの付いていない行に、原価割れが混ざっていないこと。
    # 値引率の異常（discount_error）は**わざと**原価割れにしているので除外する。
    labelled = anomaly_labels.select("受注番号", "明細番号").unique()
    normal = transactions.join(labelled, on=["受注番号", "明細番号"], how="anti")
    bad_margin = normal.filter((pl.col("数量") > 0) & (pl.col("粗利") < 0)).height

    return [
        Check(
            "前年同月比の外し率",
            f"{yoy:.1%}（{n_months}ヶ月で評価）",
            _in_range(yoy, YOY_WAPE_RANGE),
            f"許容 {YOY_WAPE_RANGE[0]:.0%}〜{YOY_WAPE_RANGE[1]:.0%}。"
            "低すぎると Excel に勝てず、機械学習の題材にならない",
        ),
        Check(
            "離反顧客の割合",
            f"{churn:.1%}",
            _in_range(churn, CHURN_RATIO_RANGE),
            f"許容 {CHURN_RATIO_RANGE[0]:.0%}〜{CHURN_RATIO_RANGE[1]:.0%}。"
            "少なすぎると分類の学習が成立しない",
        ),
        Check(
            "異常の混入率",
            f"{anomaly_ratio:.2%}（{anomaly_rows} 行）",
            _in_range(anomaly_ratio, ANOMALY_RATIO_RANGE),
            f"許容 {ANOMALY_RATIO_RANGE[0]:.1%}〜{ANOMALY_RATIO_RANGE[1]:.0%}",
        ),
        Check(
            "大型案件の割合",
            f"{large:.2%}",
            _in_range(large, LARGE_DEAL_RATIO_RANGE),
            f"許容 {LARGE_DEAL_RATIO_RANGE[0]:.1%}〜{LARGE_DEAL_RATIO_RANGE[1]:.0%}。"
            "月次の着地を振らせる要因",
        ),
        Check(
            "受注の無い部署×月",
            f"{empty_months} 組",
            empty_months == 0,
            "0であること。月次の着地予測は部署×月が予測の単位なので、穴があると成立しない",
        ),
        Check(
            "顧客あたり月間受注回数",
            f"{frequency:.1f} 回",
            frequency >= MIN_ORDERS_PER_CUSTOMER_MONTH,
            f"下限 {MIN_ORDERS_PER_CUSTOMER_MONTH:.1f} 回。"
            "低すぎると「たまたま今月無かった」と「離反」が区別できない",
        ),
        Check(
            "通常取引の原価割れ",
            f"{bad_margin} 行",
            bad_margin == 0,
            "0件であること。異常ラベル付きの行は、わざと原価割れにしているので除く",
        ),
    ]


def format_checks(checks: list[Check]) -> str:
    """検査結果を人が読める形に整える。"""
    width = max(len(c.name) for c in checks)
    lines = []
    for check in checks:
        mark = "OK  " if check.passed else "NG  "
        lines.append(f"  {mark}{check.name.ljust(width)}  {check.value}")
        if not check.passed:
            lines.append(f"        → {check.note}")
    return "\n".join(lines)
