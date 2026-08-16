"""origin 側と target 側を結合して学習/推論用の特徴量行列を組み立てる。

設計の中心は次の一点に尽きる。

    **origin 行と target 行を、必ず horizon 日ずらして結合する。**

この結合を1か所に閉じ込めているため、「うっかり当日の実績を特徴量に混ぜる」
という時系列で最も起きやすい事故が構造的に起こらない。学習時と推論時で
同じ関数を通すので、学習・推論間の特徴量の食い違い（training-serving skew）も
生じない。

前提としている業務条件:

    価格と販促は「計画値」として予測時点で確定している。

小売の需要予測では販促計画・価格改定は数週間前に決まっているのが通常で、
この前提は実務的に妥当である。もし予測時点で価格が未確定な案件であれば、
価格自体を別モデルで予測するか、特徴量から外す必要がある。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from demand_forecast.config import FeatureConfig
from demand_forecast.features.calendar import add_calendar_features
from demand_forecast.features.lags import (
    add_origin_features,
    all_same_dow_columns,
    same_dow_columns,
)
from demand_forecast.polars_utils import as_date

# 特徴量ではない（識別・目的変数の）カラム
FEATURE_METADATA_KEYS: tuple[str, ...] = (
    "date",
    "origin_date",
    "store_id",
    "sku_id",
    "y",
)

_KEY_COLS = ["store_id", "sku_id"]


@dataclass(frozen=True)
class SeriesEncoder:
    """店舗ID・商品IDを整数コードに変換する。

    LightGBM のカテゴリ特徴量として渡すために整数化する。学習時に作った
    対応表をモデルと一緒に保存し、推論時も同じ対応表を使う。
    未知のIDは ``-1``（学習時に存在しなかったカテゴリ）に落とす。
    """

    store_to_code: dict[str, int]
    sku_to_code: dict[str, int]

    @classmethod
    def fit(cls, df: pl.DataFrame) -> SeriesEncoder:
        stores = sorted(df.get_column("store_id").unique().to_list())
        skus = sorted(df.get_column("sku_id").unique().to_list())
        return cls(
            store_to_code={s: i for i, s in enumerate(stores)},
            sku_to_code={s: i for i, s in enumerate(skus)},
        )

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("store_id")
            .replace_strict(self.store_to_code, default=-1, return_dtype=pl.Int32)
            .alias("feat_store_code"),
            pl.col("sku_id")
            .replace_strict(self.sku_to_code, default=-1, return_dtype=pl.Int32)
            .alias("feat_sku_code"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"store_to_code": self.store_to_code, "sku_to_code": self.sku_to_code}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SeriesEncoder:
        return cls(
            store_to_code=dict(payload["store_to_code"]),
            sku_to_code=dict(payload["sku_to_code"]),
        )


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """0除算を避けた比率。分母が 0 または欠損なら null を返す。"""
    return (
        pl.when(denominator.is_not_null() & (denominator > 0))
        .then(numerator / denominator)
        .otherwise(None)
    )


def _assemble(
    origin_feats: pl.DataFrame,
    targets: pl.DataFrame,
    *,
    horizon: int,
    feat_cfg: FeatureConfig,
) -> pl.DataFrame:
    """origin 特徴量を horizon 日ずらして target 行に結合する。"""
    rename_map = same_dow_columns(horizon)
    unused_dow_cols = [c for c in all_same_dow_columns() if c not in rename_map]

    origin_side = (
        origin_feats.drop(unused_dow_cols)
        .rename(rename_map)
        .with_columns(pl.col("date").dt.offset_by(f"{horizon}d").alias("_target_date"))
    )
    org_cols = [c for c in origin_side.columns if c.startswith("org_")]
    origin_side = origin_side.select(
        [pl.col("date").alias("origin_date"), pl.col("_target_date"), *_KEY_COLS, *org_cols]
    )

    joined = targets.join(
        origin_side,
        left_on=["date", *_KEY_COLS],
        right_on=["_target_date", *_KEY_COLS],
        how="inner",
    )

    short_window = min(feat_cfg.rolling_windows)
    long_window = max(feat_cfg.rolling_windows)

    joined = joined.with_columns(
        pl.lit(horizon, dtype=pl.Int16).alias("feat_horizon"),
        pl.col("promo_flag").cast(pl.Int8).alias("feat_promo_flag"),
        pl.col("price").cast(pl.Float64).alias("feat_price"),
        # 「絶対価格」より「いつもと比べて高いか安いか」のほうが需要を説明する
        _safe_ratio(pl.col("price"), pl.col("org_price_mean_28")).alias("feat_price_ratio_28"),
        _safe_ratio(pl.col("price"), pl.col("org_price")).alias("feat_price_vs_origin"),
        # 直近水準が長期水準からどれだけ離れているか（トレンドの代理指標）
        _safe_ratio(
            pl.col(f"org_roll_mean_{short_window}"),
            pl.col(f"org_roll_mean_{long_window}"),
        ).alias("feat_short_long_ratio"),
        # 変動の大きさ（在庫の安全余裕を測るのに効く）
        _safe_ratio(
            pl.col(f"org_roll_std_{long_window}"),
            pl.col(f"org_roll_mean_{long_window}"),
        ).alias("feat_cv_long"),
    ).drop("price", "promo_flag")

    return add_calendar_features(joined, fourier_order=feat_cfg.fourier_yearly_order)


def build_training_frame(
    df: pl.DataFrame,
    feat_cfg: FeatureConfig,
    *,
    target_col: str = "units_sold",
) -> pl.DataFrame:
    """horizon 1〜H のすべてについて学習行を作る。

    horizon ごとに別モデルを建てる（direct 戦略）のではなく、
    horizon を特徴量として持つ単一モデルにまとめている。理由は
    ``docs/adr/0004-horizon-aware-features.md`` を参照。

    Args:
        df: 需要データ（検証済み）。
        feat_cfg: 特徴量設定。
        target_col: 目的変数のカラム名。

    Returns:
        1行 = (系列, target 日, horizon) の学習用フレーム。目的変数は ``y``。
    """
    origin_feats = add_origin_features(df, lags=feat_cfg.lags, windows=feat_cfg.rolling_windows)
    targets = df.select(
        ["date", *_KEY_COLS, "price", "promo_flag", pl.col(target_col).cast(pl.Float64).alias("y")]
    )

    frames = [
        _assemble(origin_feats, targets, horizon=h, feat_cfg=feat_cfg)
        for h in range(1, feat_cfg.horizon + 1)
    ]
    combined = pl.concat(frames, how="vertical")

    # origin 側の履歴が全く無い行は学習に使えない
    lag_col = f"org_lag_{min(feat_cfg.lags)}"
    return combined.filter(pl.col(lag_col).is_not_null()).sort(["date", *_KEY_COLS, "feat_horizon"])


def build_inference_frame(
    history: pl.DataFrame,
    future: pl.DataFrame,
    feat_cfg: FeatureConfig,
    *,
    target_col: str = "units_sold",
) -> pl.DataFrame:
    """推論用の特徴量行列を組み立てる。

    Args:
        history: origin 日までの実績（``date``/``store_id``/``sku_id``/
            ``units_sold``/``price``/``promo_flag``）。
        future: 予測対象日の計画値（``date``/``store_id``/``sku_id``/
            ``price``/``promo_flag``）。
        feat_cfg: 学習時と同じ特徴量設定。
        target_col: 実績カラム名。

    Returns:
        ``future`` の各行に対応する特徴量行列。``future`` の行のうち
        horizon が範囲外のものは落とされる。

    Raises:
        ValueError: ``history`` が空、または horizon が 1〜H の範囲外の
            予測対象日が含まれる場合。
    """
    if history.is_empty():
        msg = "history が空です。特徴量を作るには実績が必要です。"
        raise ValueError(msg)

    origin_date = as_date(history.get_column("date").max())
    origin_feats = add_origin_features(
        history, lags=feat_cfg.lags, windows=feat_cfg.rolling_windows
    ).filter(pl.col("date") == origin_date)

    targets = future.select(
        [
            "date",
            *_KEY_COLS,
            "price",
            "promo_flag",
            pl.lit(None, dtype=pl.Float64).alias("y"),
        ]
    )

    horizons = (
        targets.select((pl.col("date") - pl.lit(origin_date)).dt.total_days().alias("h"))
        .get_column("h")
        .unique()
        .sort()
        .to_list()
    )
    invalid = [h for h in horizons if not 1 <= h <= feat_cfg.horizon]
    if invalid:
        msg = (
            f"予測対象日の horizon が範囲外です: {invalid}。"
            f"origin={origin_date} からの 1〜{feat_cfg.horizon} 日先のみ対応しています。"
        )
        raise ValueError(msg)

    frames = [
        _assemble(
            origin_feats,
            targets.filter(
                (pl.col("date") - pl.lit(origin_date)).dt.total_days() == h,
            ),
            horizon=int(h),
            feat_cfg=feat_cfg,
        )
        for h in horizons
    ]
    return pl.concat(frames, how="vertical").sort(["date", *_KEY_COLS])


def feature_columns(frame: pl.DataFrame) -> list[str]:
    """特徴量として使うカラム名を返す（メタデータ列を除いたもの）。"""
    return [c for c in frame.columns if c not in FEATURE_METADATA_KEYS]


def categorical_features(columns: list[str]) -> list[str]:
    """LightGBM にカテゴリとして渡すカラム名を返す。"""
    return [c for c in ("feat_store_code", "feat_sku_code") if c in columns]
