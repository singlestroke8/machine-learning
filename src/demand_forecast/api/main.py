"""推論API本体。

設計方針:

- モデルはプロセス起動時に1度だけ読み込む（リクエストごとの読み込みは論外）
- モデルが無い状態でも起動はする。``/health`` が ``model_loaded: false`` を
  返すので、オーケストレータが正しく異常と判断できる。起動時に落とすと
  クラッシュループになり、原因がログから追いにくい。
- 特徴量生成はサービス層（``models.predict``）に委譲し、
  ここでは HTTP の関心事だけを扱う。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException, status

from demand_forecast import __version__
from demand_forecast.api.schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfoResponse,
)
from demand_forecast.config import ApiSettings
from demand_forecast.logging_utils import configure_logging, get_logger
from demand_forecast.models.estimator import ForecastArtifact
from demand_forecast.models.predict import InsufficientHistoryError, forecast

logger = get_logger(__name__)

# プロセス内に1つだけ持つモデル。テストからも差し替えられるよう辞書にしている。
_state: dict[str, Any] = {"artifact": None}


def get_artifact() -> ForecastArtifact | None:
    """読み込み済みモデルを返す（未読み込みなら None）。"""
    artifact = _state.get("artifact")
    return artifact if isinstance(artifact, ForecastArtifact) else None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """起動時にモデルを読み込み、終了時に解放する。"""
    settings = ApiSettings()
    configure_logging(settings.log_level)
    try:
        _state["artifact"] = ForecastArtifact.load(settings.model_path)
        logger.info("モデルを読み込みました: %s", settings.model_path)
    except (FileNotFoundError, ValueError) as exc:
        _state["artifact"] = None
        logger.error("モデルの読み込みに失敗しました: %s", exc)

    yield

    _state["artifact"] = None
    logger.info("モデルを解放しました")


app = FastAPI(
    title="Demand Forecast API",
    description=(
        "店舗×商品の日次需要を、予測区間つきで返す推論API。\n\n"
        "実績の系列と、予測対象日の価格・販促計画を送ると、"
        "1〜14日先の需要を中央値と 80% 予測区間で返します。"
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    """死活監視用。モデルが使える状態かどうかまで返す。"""
    artifact = get_artifact()
    return HealthResponse(
        status="ok" if artifact is not None else "degraded",
        model_loaded=artifact is not None,
        horizon=artifact.feature_config.horizon if artifact else None,
    )


@app.get("/model", response_model=ModelInfoResponse, tags=["monitoring"])
def model_info() -> ModelInfoResponse:
    """稼働中モデルの素性を返す。

    「今どのモデルが動いているか」を後から確認できないと、
    精度の議論が再現できない。

    Raises:
        HTTPException: モデルが読み込まれていない場合 (503)。
    """
    artifact = _require_artifact()
    meta = artifact.metadata
    cv_summary = meta.get("cv_summary") or {}
    return ModelInfoResponse(
        trained_at=meta.get("trained_at"),
        horizon=artifact.feature_config.horizon,
        quantiles=sorted(artifact.model.models),
        n_features=len(artifact.model.feature_names),
        n_series=meta.get("n_series"),
        data_start=meta.get("data_start"),
        data_end=meta.get("data_end"),
        cv_wape=cv_summary.get("wape_mean"),
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["inference"])
def create_forecast(request: ForecastRequest) -> ForecastResponse:
    """需要予測を返す。

    Raises:
        HTTPException: モデル未読み込み (503)、入力が推論条件を満たさない (422)、
            推論中の想定外エラー (500)。
    """
    artifact = _require_artifact()

    history = pl.DataFrame(
        {
            "date": [p.date for p in request.history],
            "store_id": [request.store_id] * len(request.history),
            "sku_id": [request.sku_id] * len(request.history),
            "units_sold": [p.units_sold for p in request.history],
            "price": [p.price for p in request.history],
            "promo_flag": [p.promo_flag for p in request.history],
        },
        schema={
            "date": pl.Date,
            "store_id": pl.Utf8,
            "sku_id": pl.Utf8,
            "units_sold": pl.Int32,
            "price": pl.Float64,
            "promo_flag": pl.Int8,
        },
    )
    future = pl.DataFrame(
        {
            "date": [p.date for p in request.future],
            "store_id": [request.store_id] * len(request.future),
            "sku_id": [request.sku_id] * len(request.future),
            "price": [p.price for p in request.future],
            "promo_flag": [p.promo_flag for p in request.future],
        },
        schema={
            "date": pl.Date,
            "store_id": pl.Utf8,
            "sku_id": pl.Utf8,
            "price": pl.Float64,
            "promo_flag": pl.Int8,
        },
    )

    try:
        result = forecast(artifact, history, future)
    except (InsufficientHistoryError, ValueError) as exc:
        # 入力の問題はクライアント側で直せるので、内容をそのまま返す
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except Exception as exc:  # pragma: no cover - 想定外の防波堤
        logger.exception("推論中に想定外のエラーが発生しました")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="推論中にサーバ内部エラーが発生しました。",
        ) from exc

    rows = result.to_dicts()
    return ForecastResponse(
        store_id=request.store_id,
        sku_id=request.sku_id,
        origin_date=rows[0]["origin_date"],
        lower_quantile=rows[0]["lower_quantile"],
        upper_quantile=rows[0]["upper_quantile"],
        forecasts=[
            ForecastPoint(
                date=row["date"],
                horizon=row["horizon"],
                point=row["point"],
                lower=row["lower"],
                upper=row["upper"],
            )
            for row in rows
        ],
        model_trained_at=artifact.metadata.get("trained_at"),
    )


def _require_artifact() -> ForecastArtifact:
    """モデルを取得する。未読み込みなら 503 を返す。

    Raises:
        HTTPException: モデルが読み込まれていない場合 (503)。
    """
    artifact = get_artifact()
    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="モデルが読み込まれていません。学習済みモデルを配置して再起動してください。",
        )
    return artifact
