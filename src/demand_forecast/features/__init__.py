"""特徴量エンジニアリング。

このパッケージの最重要な責務は「予測時点で本当に手に入る情報だけを使う」
ことを構造的に保証する点にある。詳細は
``docs/adr/0004-horizon-aware-features.md`` を参照。
"""

from demand_forecast.features.calendar import (
    add_calendar_features,
    japanese_holiday_flags,
)
from demand_forecast.features.lags import add_origin_features, same_dow_offset
from demand_forecast.features.pipeline import (
    FEATURE_METADATA_KEYS,
    SeriesEncoder,
    build_inference_frame,
    build_training_frame,
    categorical_features,
    feature_columns,
)

__all__ = [
    "FEATURE_METADATA_KEYS",
    "SeriesEncoder",
    "add_calendar_features",
    "add_origin_features",
    "build_inference_frame",
    "build_training_frame",
    "categorical_features",
    "feature_columns",
    "japanese_holiday_flags",
    "same_dow_offset",
]
