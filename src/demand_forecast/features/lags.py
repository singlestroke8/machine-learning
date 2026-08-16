"""origin（予測基準日）時点で確定している系列特徴量。

用語の定義:

- **origin**: 予測を行う基準日。この日の実績までが手元にある。
- **target**: 予測対象日。``target = origin + horizon``。

このモジュールが作るのは「origin 時点の特徴量」だけである。
target 側の情報は一切参照しない。origin 行と target 行の結合は
``pipeline.py`` が担当し、そこで horizon 分だけずらして結合することで
リークが起きない構造になっている。

命名規約: ``org_lag_k`` は **origin を1日目として数えて k 日前** の実績。
つまり ``org_lag_1`` は origin 当日の実績そのものを指す。
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

DEFAULT_GROUP_COLS: tuple[str, ...] = ("store_id", "sku_id")

# 同一曜日の傾向をとらえるために遡る週数
_SAME_DOW_WEEKS = 4

# Polars の曜日表現（月曜=1 〜 日曜=7）
_WEEKDAYS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)

# 曜日ごとの集計に使う作業列（戻り値には残さない）
_DOW_COL = "_dow"
_DOW_MEAN_COL = "_dow_mean"


def same_dow_columns() -> list[str]:
    """曜日ごとの中間カラム名を返す（結合後に落とすために使う）。"""
    return [f"org_dowlast_w{d}" for d in _WEEKDAYS] + [f"org_dowmean_w{d}" for d in _WEEKDAYS]


def select_target_dow(prefix: str, date_col: str = "date") -> pl.Expr:
    """target の曜日に対応する列を、行ごとに選ぶ式を返す。

    horizon から固定のオフセットを計算する方式は、祝日を挟むと曜日がずれる。
    target の曜日そのものを見て選べば、暦日でも営業日でも正確になる。
    """
    target_dow = pl.col(date_col).dt.weekday()
    return pl.coalesce([pl.when(target_dow == d).then(pl.col(f"{prefix}{d}")) for d in _WEEKDAYS])


def add_origin_features(
    df: pl.DataFrame,
    *,
    lags: Sequence[int],
    windows: Sequence[int],
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    date_col: str = "date",
    target_col: str = "units_sold",
    price_col: str = "price",
    promo_col: str = "promo_flag",
) -> pl.DataFrame:
    """origin 時点で参照可能な特徴量を付与する。

    入力は系列ごとに日付の欠落がない日次データであることを前提とする
    （``loaders.validate_demand_frame`` で検証済みであること）。
    欠落があると ``shift`` の意味が日数からずれてしまうため。

    Args:
        df: 需要データ。
        lags: origin からのラグ（1 = origin 当日）。単位はステップ。
        windows: 移動集計の窓幅（ステップ数）。origin 当日を含む。
        group_cols: 系列を識別するカラム。
        date_col: 日付カラム名。
        target_col: 実績カラム名。
        price_col: 価格カラム名。
        promo_col: 販促フラグカラム名。

    Returns:
        ``org_`` 接頭辞の特徴量を追加した DataFrame。
    """
    group = list(group_cols)
    y = pl.col(target_col).cast(pl.Float64)

    # 曜日ごとの集計に使う作業列。同じ曜日だけを集めた並びの上で移動平均をとる
    out = df.sort([*group, date_col]).with_columns(pl.col(date_col).dt.weekday().alias(_DOW_COL))
    out = out.with_columns(
        y.rolling_mean(_SAME_DOW_WEEKS).over([*group, _DOW_COL]).alias(_DOW_MEAN_COL)
    )

    exprs: list[pl.Expr] = []

    # --- ラグ ---
    for k in lags:
        exprs.append(y.shift(k - 1).over(group).alias(f"org_lag_{k}"))

    # --- 移動集計（origin 当日を含む窓） ---
    for w in windows:
        exprs.append(y.rolling_mean(window_size=w).over(group).alias(f"org_roll_mean_{w}"))
        exprs.append(y.rolling_std(window_size=w).over(group).alias(f"org_roll_std_{w}"))

    # --- 曜日ごとの直近実績 ---
    # 「origin から何ステップ前が target と同じ曜日か」を固定の数で表すことはできない。
    # 暦日なら7ステップ前だが、営業日軸では祝日を挟んだ週だけ間隔が縮む。
    # そこで曜日ごとに「その曜日で最後に観測した値」を持たせ、
    # pipeline 側が target の曜日に応じて該当する1本を選ぶ。祝日があっても正確になる。
    for weekday in _WEEKDAYS:
        on_weekday = pl.col(_DOW_COL) == weekday
        # その曜日の直近実績（季節性ナイーブ予測そのもの）
        exprs.append(
            pl.when(on_weekday)
            .then(y)
            .otherwise(None)
            .forward_fill()
            .over(group)
            .alias(f"org_dowlast_w{weekday}")
        )
        # その曜日の直近4回の平均（単発のブレをならしたもの）
        exprs.append(
            pl.when(on_weekday)
            .then(pl.col(_DOW_MEAN_COL))
            .otherwise(None)
            .forward_fill()
            .over(group)
            .alias(f"org_dowmean_w{weekday}")
        )

    # --- 価格・販促の直近状況 ---
    exprs.extend(
        [
            pl.col(price_col).alias("org_price"),
            pl.col(price_col).rolling_mean(window_size=28).over(group).alias("org_price_mean_28"),
            pl.col(promo_col)
            .cast(pl.Float64)
            .rolling_mean(window_size=28)
            .over(group)
            .alias("org_promo_rate_28"),
            (y == 0)
            .cast(pl.Float64)
            .rolling_mean(window_size=28)
            .over(group)
            .alias("org_zero_rate_28"),
        ]
    )

    return out.with_columns(exprs).drop(_DOW_COL, _DOW_MEAN_COL)
