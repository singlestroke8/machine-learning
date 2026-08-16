"""推論のサービス層。

API と CLI の両方から呼ばれる。ここで「学習時とまったく同じ特徴量生成を
通す」ことを保証し、API 側は HTTP の入出力だけに集中できるようにしている。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from demand_forecast.features.pipeline import build_inference_frame
from demand_forecast.logging_utils import get_logger
from demand_forecast.models.estimator import ForecastArtifact
from demand_forecast.polars_utils import as_date

logger = get_logger(__name__)

# 移動平均・同一曜日平均が意味のある値になるまでに必要な日数。
# これを下回っても推論自体は動く（欠損として扱われる）が、精度は保証できない。
RECOMMENDED_HISTORY_DAYS = 28

_KEY_COLS = ["store_id", "sku_id"]


class InsufficientHistoryError(ValueError):
    """履歴が短すぎて推論できない場合に送出する。"""


def _check_single_origin(history: pl.DataFrame) -> dt.date:
    """全系列の履歴末尾が揃っていることを確認し、origin 日を返す。

    系列ごとに履歴の長さが違うと、同じリクエストの中で origin が
    バラバラになり、horizon の意味が系列ごとに変わってしまう。
    静かに混ざるとデバッグが難しいので、ここで弾く。

    Raises:
        InsufficientHistoryError: 系列間で履歴末尾が揃っていない場合。
    """
    per_series = history.group_by(_KEY_COLS).agg(pl.col("date").max().alias("last_date"))
    last_dates = per_series.get_column("last_date").unique().to_list()
    if len(last_dates) > 1:
        msg = (
            "系列ごとに履歴の最終日が異なります。"
            f" 検出された最終日: {sorted(str(d) for d in last_dates)}。"
            " すべての系列を同じ日まで揃えてください。"
        )
        raise InsufficientHistoryError(msg)
    return as_date(last_dates[0])


def forecast(
    artifact: ForecastArtifact,
    history: pl.DataFrame,
    future: pl.DataFrame,
) -> pl.DataFrame:
    """需要予測を実行する。

    Args:
        artifact: 学習済みモデル一式。
        history: origin 日までの実績。
        future: 予測対象日の計画値（価格・販促）。

    Returns:
        ``date``/``store_id``/``sku_id``/``horizon``/``point``/``lower``/``upper``
        を持つ DataFrame。``future`` の並びとは無関係に日付順で返す。

    Raises:
        InsufficientHistoryError: 履歴が空、または系列間で末尾が揃っていない場合。
    """
    if history.is_empty():
        msg = "履歴が空です。"
        raise InsufficientHistoryError(msg)

    origin_date = _check_single_origin(history)
    history_days = history.get_column("date").n_unique()
    if history_days < RECOMMENDED_HISTORY_DAYS:
        logger.warning(
            "履歴が %d 日しかありません（推奨 %d 日以上）。長期の特徴量が欠損し、精度が落ちます。",
            history_days,
            RECOMMENDED_HISTORY_DAYS,
        )

    frame = build_inference_frame(history, future, artifact.feature_config)
    frame = artifact.encoder.transform(frame)
    predictions = artifact.model.predict(frame)

    quantiles = sorted(predictions)
    lower_q, upper_q = quantiles[0], quantiles[-1]

    return (
        frame.select(["date", *_KEY_COLS, "feat_horizon"])
        .with_columns(
            pl.Series("point", np.round(predictions[0.5], 2), dtype=pl.Float64),
            pl.Series("lower", np.round(predictions[lower_q], 2), dtype=pl.Float64),
            pl.Series("upper", np.round(predictions[upper_q], 2), dtype=pl.Float64),
            pl.lit(lower_q, dtype=pl.Float64).alias("lower_quantile"),
            pl.lit(upper_q, dtype=pl.Float64).alias("upper_quantile"),
            pl.lit(origin_date).alias("origin_date"),
        )
        .rename({"feat_horizon": "horizon"})
        .sort(["date", *_KEY_COLS])
    )
