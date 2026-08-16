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
        lags: origin からのラグ日数（1 = origin 当日）。
        windows: 移動集計の窓幅（日数）。origin 当日を含む。
        group_cols: 系列を識別するカラム。
        date_col: 日付カラム名。
        target_col: 実績カラム名。
        price_col: 価格カラム名。
        promo_col: 販促フラグカラム名。

    Returns:
        ``org_`` 接頭辞の特徴量を追加した DataFrame。
    """
    group = list(group_cols)
    out = df.sort([*group, date_col])
    y = pl.col(target_col).cast(pl.Float64)

    exprs: list[pl.Expr] = []

    # --- ラグ ---
    for k in lags:
        exprs.append(y.shift(k - 1).over(group).alias(f"org_lag_{k}"))

    # --- 移動集計（origin 当日を含む窓） ---
    for w in windows:
        exprs.append(y.rolling_mean(window_size=w).over(group).alias(f"org_roll_mean_{w}"))
        exprs.append(y.rolling_std(window_size=w).over(group).alias(f"org_roll_std_{w}"))

    # --- 同一曜日の実績 ---
    # r = 0..6 は「origin から何日前が target と同じ曜日か」に対応する。
    # どの r を使うかは horizon で決まるため、ここでは 7 通りすべてを作り、
    # pipeline 側で該当する 1 本だけを選ぶ。
    for r in range(7):
        # 直近の同一曜日（季節性ナイーブ予測そのもの）
        exprs.append(y.shift(r).over(group).alias(f"org_dowlast_r{r}"))
        # 直近4回の同一曜日の平均（単発のブレをならしたもの）
        shifts = [y.shift(r + 7 * w) for w in range(_SAME_DOW_WEEKS)]
        mean_expr = sum(shifts[1:], start=shifts[0]) / _SAME_DOW_WEEKS
        exprs.append(mean_expr.over(group).alias(f"org_dowmean_r{r}"))

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

    return out.with_columns(exprs)


def same_dow_offset(horizon: int) -> int:
    """target と同じ曜日になる直近の日が origin の何日前かを返す。

    target の曜日は origin の曜日から ``horizon`` 日進んだもの。したがって
    origin 以前で target と同じ曜日になる直近の日は ``(-horizon) % 7`` 日前。
    """
    return (-horizon) % 7


def same_dow_columns(horizon: int) -> dict[str, str]:
    """horizon に対応する同一曜日カラムの、旧名 -> 新名の対応を返す。"""
    r = same_dow_offset(horizon)
    return {
        f"org_dowlast_r{r}": "org_target_dow_last",
        f"org_dowmean_r{r}": "org_target_dow_mean",
    }


def all_same_dow_columns() -> list[str]:
    """同一曜日カラム名を全通り返す（不要分の削除に使う）。"""
    return [f"org_dowlast_r{r}" for r in range(7)] + [f"org_dowmean_r{r}" for r in range(7)]
