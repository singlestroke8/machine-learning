"""取引明細を、需要予測が扱える日次の系列データに集計する。

明細（1行 = 1受注の1品目）と、学習に使う系列データ（1行 = 系列 × 1ステップ）は
形が違う。ここがその変換を担う。

変換で決めていること:

1. **系列の粒度**: 既定は「部署 × 品名」。東京1拠点なので拠点は系列識別に使えず、
   顧客×品名では疎すぎる（顧客が特定商品を毎営業日買うことはない）。
   部署×品名なら30系列となり、日次予測が成り立つ密度になる。
2. **時間軸**: 営業日のみ。土日祝は法人取引が発生しないので、
   ゼロの行を並べるより営業日を1ステップとして数えるほうが素直
   （``docs/adr/0011-business-day-timeline.md``）。
3. **受注が無かった営業日**: 数量0の行として補う。落とすと時間軸に穴が空き、
   ラグが「何ステップ前か」を意味しなくなる。
4. **価格**: 数量で重み付けした平均販売単価。受注が無かった日は
   直近の単価を持ち越す（価格表は売れなくても存在するため）。
5. **販促フラグ**: 平均値引率がしきい値以上なら1。B2B に「販促」は無いが、
   「大口値引き案件があった日」は需要を押し上げる要因として同じ働きをする。
"""

from __future__ import annotations

import polars as pl

from demand_forecast.data.loaders import validate_demand_frame
from demand_forecast.features.calendar import business_days
from demand_forecast.logging_utils import get_logger
from demand_forecast.polars_utils import as_date, as_float

logger = get_logger(__name__)

#: 系列を識別する明細側のカラム（store_id / sku_id に対応づける）
DEFAULT_SERIES_KEYS: tuple[str, str] = ("部署", "品名")

#: 販促フラグを立てる値引率のしきい値。
#: 固定値にしているのは、分位点などデータ全体の統計から決めると
#: 検証期間の情報が学習期間に漏れるため。
DISCOUNT_PROMO_THRESHOLD = 0.10


def aggregate_transactions(
    transactions: pl.DataFrame,
    *,
    series_keys: tuple[str, str] = DEFAULT_SERIES_KEYS,
    date_col: str = "受注日",
    quantity_col: str = "数量",
    amount_col: str = "販売金額",
    discount_col: str = "値引率",
    promo_threshold: float = DISCOUNT_PROMO_THRESHOLD,
) -> pl.DataFrame:
    """取引明細を営業日ベースの系列データに集計する。

    Args:
        transactions: 取引明細。
        series_keys: 系列を識別する2列。それぞれ ``store_id``/``sku_id`` になる。
        date_col: 受注日のカラム名。
        quantity_col: 数量のカラム名。
        amount_col: 販売金額のカラム名。
        discount_col: 値引率のカラム名。
        promo_threshold: 販促フラグを立てる平均値引率のしきい値。

    Returns:
        ``DEMAND_SCHEMA`` に沿った DataFrame。営業日 × 系列の完全な格子になり、
        受注が無かった営業日は ``units_sold = 0`` で埋まる。

    Raises:
        KeyError: 必要なカラムが明細に無い場合。
        ValueError: 明細が空、または営業日以外の受注が含まれる場合。
    """
    required = {date_col, quantity_col, amount_col, discount_col, *series_keys}
    missing = required - set(transactions.columns)
    if missing:
        msg = f"明細に必要なカラムがありません: {sorted(missing)}"
        raise KeyError(msg)
    if transactions.is_empty():
        msg = "明細が空です。"
        raise ValueError(msg)

    store_key, sku_key = series_keys

    # --- 1. 系列 × 受注日で集計する ---
    daily = transactions.group_by([date_col, store_key, sku_key]).agg(
        pl.col(quantity_col).sum().alias("units_sold"),
        pl.col(amount_col).sum().alias("_amount"),
        # 数量で重み付けした平均値引率。大口案件の値引きを正しく反映させる
        ((pl.col(discount_col) * pl.col(quantity_col)).sum() / pl.col(quantity_col).sum()).alias(
            "_discount"
        ),
    )

    # --- 2. 営業日 × 系列の完全な格子を作る ---
    observed_dates = sorted(set(daily.get_column(date_col).to_list()))
    timeline = business_days(as_date(min(observed_dates)), as_date(max(observed_dates)))

    off_calendar = sorted(set(observed_dates) - set(timeline))
    if off_calendar:
        msg = (
            f"営業日以外の受注が {len(off_calendar)} 件あります: "
            f"{[str(d) for d in off_calendar[:5]]}。"
            " 明細側の生成条件を確認してください。"
        )
        raise ValueError(msg)

    series = daily.select([store_key, sku_key]).unique()
    grid = pl.DataFrame({date_col: timeline}).join(series, how="cross")

    # --- 3. 受注が無かった営業日を 0 で埋める ---
    filled = (
        grid.join(daily, on=[date_col, store_key, sku_key], how="left")
        .with_columns(
            pl.col("units_sold").fill_null(0),
            pl.col("_amount").fill_null(0),
        )
        .sort([store_key, sku_key, date_col])
    )

    # --- 4. 価格と販促フラグを組み立てる ---
    group = [store_key, sku_key]
    result = (
        filled.with_columns(
            # 受注のあった日は加重平均単価。無かった日は null にしておき、後で持ち越す
            pl.when(pl.col("units_sold") > 0)
            .then(pl.col("_amount") / pl.col("units_sold"))
            .otherwise(None)
            .alias("price"),
        )
        .with_columns(
            # 価格表は売れていない日にも存在するので、直近の単価を前後から補う
            pl.col("price").forward_fill().backward_fill().over(group),
            (pl.col("_discount").fill_null(0.0) >= promo_threshold)
            .cast(pl.Int8)
            .alias("promo_flag"),
        )
        .select(
            pl.col(date_col).cast(pl.Date).alias("date"),
            pl.col(store_key).cast(pl.Utf8).alias("store_id"),
            pl.col(sku_key).cast(pl.Utf8).alias("sku_id"),
            pl.col("units_sold").cast(pl.Int32),
            pl.col("price").round(2).cast(pl.Float64),
            pl.col("promo_flag"),
        )
        .sort(["date", "store_id", "sku_id"])
    )

    logger.info(
        "集計しました: %d 明細 -> %d 行（%d 系列 × %d 営業日）",
        transactions.height,
        result.height,
        series.height,
        len(timeline),
    )
    return validate_demand_frame(result, calendar="business")


def summarize(demand: pl.DataFrame) -> dict[str, object]:
    """集計結果の検算値を返す。"""
    n_series = demand.select(["store_id", "sku_id"]).unique().height
    start = as_date(demand.get_column("date").min())
    end = as_date(demand.get_column("date").max())

    per_series = demand.group_by(["store_id", "sku_id"]).agg(
        (pl.col("units_sold") == 0).mean().alias("zero_rate")
    )
    worst = per_series.sort("zero_rate", descending=True).head(1).to_dicts()[0]

    return {
        "行数": demand.height,
        "系列数": n_series,
        "営業日数": demand.get_column("date").n_unique(),
        "期間": f"{start} 〜 {end}",
        "ゼロ率（全体）": f"{100 * as_float((demand.get_column('units_sold') == 0).mean()):.1f}%",
        "ゼロ率（最悪の系列）": (
            f"{100 * as_float(worst['zero_rate']):.1f}%（{worst['store_id']} / {worst['sku_id']}）"
        ),
        "数量の平均": round(as_float(demand.get_column("units_sold").mean()), 2),
        "販促日の割合": f"{100 * as_float(demand.get_column('promo_flag').mean()):.1f}%",
    }
