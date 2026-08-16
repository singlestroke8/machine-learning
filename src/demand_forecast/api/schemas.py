"""API の入出力スキーマ。

バリデーションは「モデルに渡す前に落ちるべきものを落とす」ことに徹している。
特に日付の連続性は重要で、履歴に穴があるとラグ特徴量が静かにずれ、
エラーにならないまま誤った予測を返してしまう。
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Self

from pydantic import BaseModel, Field, model_validator


class HistoryPoint(BaseModel):
    """origin 日までの実績1日分。"""

    date: dt.date
    units_sold: Annotated[int, Field(ge=0, description="販売数量")]
    price: Annotated[float, Field(gt=0, description="実売価格")]
    promo_flag: Annotated[int, Field(ge=0, le=1, description="販促の有無 (0/1)")]


class FuturePoint(BaseModel):
    """予測対象日1日分の計画値。

    価格と販促は予測時点で確定している計画値として受け取る。
    """

    date: dt.date
    price: Annotated[float, Field(gt=0, description="計画売価")]
    promo_flag: Annotated[int, Field(ge=0, le=1, description="販促計画 (0/1)")]


class ForecastRequest(BaseModel):
    """需要予測リクエスト（1系列ぶん）。"""

    store_id: Annotated[str, Field(min_length=1, description="店舗ID")]
    sku_id: Annotated[str, Field(min_length=1, description="商品ID")]
    history: Annotated[
        list[HistoryPoint],
        Field(min_length=1, description="実績の系列。日付が1日刻みで連続していること"),
    ]
    future: Annotated[
        list[FuturePoint],
        Field(min_length=1, description="予測対象日の計画値"),
    ]

    @model_validator(mode="after")
    def _validate_dates(self) -> Self:
        history_dates = [p.date for p in self.history]
        if len(set(history_dates)) != len(history_dates):
            msg = "history の日付が重複しています。"
            raise ValueError(msg)

        ordered = sorted(history_dates)
        expected_span = (ordered[-1] - ordered[0]).days + 1
        if expected_span != len(ordered):
            msg = (
                f"history の日付が1日刻みで連続していません "
                f"({ordered[0]}〜{ordered[-1]} に対して {len(ordered)} 件)。"
            )
            raise ValueError(msg)

        future_dates = [p.date for p in self.future]
        if len(set(future_dates)) != len(future_dates):
            msg = "future の日付が重複しています。"
            raise ValueError(msg)

        origin = ordered[-1]
        past = sorted(d for d in future_dates if d <= origin)
        if past:
            msg = f"future に origin ({origin}) 以前の日付が含まれています: {past}"
            raise ValueError(msg)
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "store_id": "S01",
                    "sku_id": "SKU01",
                    "history": [
                        {
                            "date": "2026-06-01",
                            "units_sold": 23,
                            "price": 480.0,
                            "promo_flag": 0,
                        }
                    ],
                    "future": [{"date": "2026-06-02", "price": 430.0, "promo_flag": 1}],
                }
            ]
        }
    }


class ForecastPoint(BaseModel):
    """1日分の予測結果。"""

    date: dt.date
    horizon: Annotated[int, Field(description="origin からの日数")]
    point: Annotated[float, Field(description="点予測（中央値）")]
    lower: Annotated[float, Field(description="予測区間の下限")]
    upper: Annotated[float, Field(description="予測区間の上限")]


class ForecastResponse(BaseModel):
    """需要予測レスポンス。"""

    store_id: str
    sku_id: str
    origin_date: dt.date
    lower_quantile: Annotated[float, Field(description="下限に対応する分位点")]
    upper_quantile: Annotated[float, Field(description="上限に対応する分位点")]
    forecasts: list[ForecastPoint]
    model_trained_at: str | None = Field(default=None, description="モデルの学習日時")


class HealthResponse(BaseModel):
    """ヘルスチェックの結果。"""

    status: str
    model_loaded: bool
    horizon: int | None = None


class ModelInfoResponse(BaseModel):
    """稼働中モデルの素性。"""

    trained_at: str | None
    horizon: int
    quantiles: list[float]
    n_features: int
    n_series: int | None
    data_start: str | None
    data_end: str | None
    cv_wape: float | None
