"""推論APIのテスト。

TestClient を ``with`` で使うことで lifespan（モデル読み込み）も実行される。
モデルの読み込み経路まで含めて検証しないと、
「テストは通るが本番で 503 が出る」という一番困る形の見落としが起きる。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from demand_forecast.api.main import app

ORIGIN = dt.date(2025, 6, 30)
HISTORY_DAYS = 120


@pytest.fixture
def client(artifact_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """学習済みモデルを読み込んだ状態のクライアント。"""
    monkeypatch.setenv("DFC_MODEL_PATH", str(artifact_path))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """モデルが存在しない状態のクライアント。"""
    monkeypatch.setenv("DFC_MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    with TestClient(app) as test_client:
        yield test_client


def _payload(
    demand_frame: pl.DataFrame,
    *,
    history_days: int = HISTORY_DAYS,
    forecast_days: int = 7,
) -> dict:
    """実データから正常系のリクエストを組み立てる。"""
    series = demand_frame.filter(
        (pl.col("store_id") == "S01") & (pl.col("sku_id") == "SKU01")
    ).sort("date")
    history = series.filter(pl.col("date") <= ORIGIN).tail(history_days)
    return {
        "store_id": "S01",
        "sku_id": "SKU01",
        "history": [
            {
                "date": str(row["date"]),
                "units_sold": row["units_sold"],
                "price": row["price"],
                "promo_flag": row["promo_flag"],
            }
            for row in history.to_dicts()
        ],
        "future": [
            {
                "date": str(ORIGIN + dt.timedelta(days=h)),
                "price": 450.0,
                "promo_flag": 0,
            }
            for h in range(1, forecast_days + 1)
        ],
    }


def test_health_reports_ok_when_model_is_loaded(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["horizon"] == 7


def test_health_reports_degraded_without_model(client_without_model: TestClient) -> None:
    """モデルが無くてもサーバは起動し、状態を正直に返すこと。"""
    response = client_without_model.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "model_loaded": False, "horizon": None}


def test_model_info_exposes_provenance(client: TestClient) -> None:
    response = client.get("/model")
    assert response.status_code == 200
    body = response.json()
    assert body["trained_at"] == "2026-01-01T00:00:00+00:00"
    assert body["quantiles"] == [0.1, 0.5, 0.9]
    assert body["n_features"] > 0


def test_forecast_returns_one_row_per_future_date(
    client: TestClient, demand_frame: pl.DataFrame
) -> None:
    response = client.post("/forecast", json=_payload(demand_frame, forecast_days=5))
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["origin_date"] == str(ORIGIN)
    assert len(body["forecasts"]) == 5
    assert [f["horizon"] for f in body["forecasts"]] == [1, 2, 3, 4, 5]


def test_forecast_values_are_ordered_and_non_negative(
    client: TestClient, demand_frame: pl.DataFrame
) -> None:
    """区間が下限 <= 中央値 <= 上限 で、需要が負にならないこと。"""
    response = client.post("/forecast", json=_payload(demand_frame))
    assert response.status_code == 200

    for point in response.json()["forecasts"]:
        assert point["lower"] >= 0.0
        assert point["lower"] <= point["point"] <= point["upper"]


def test_forecast_returns_503_without_model(
    client_without_model: TestClient, demand_frame: pl.DataFrame
) -> None:
    response = client_without_model.post("/forecast", json=_payload(demand_frame))
    assert response.status_code == 503


def test_history_with_gap_is_rejected(client: TestClient, demand_frame: pl.DataFrame) -> None:
    """履歴に日付の穴があったら 422 で弾くこと。"""
    payload = _payload(demand_frame)
    del payload["history"][10]

    response = client.post("/forecast", json=payload)
    assert response.status_code == 422
    assert "連続していません" in response.text


def test_future_date_before_origin_is_rejected(
    client: TestClient, demand_frame: pl.DataFrame
) -> None:
    payload = _payload(demand_frame)
    payload["future"][0]["date"] = str(ORIGIN - dt.timedelta(days=1))

    response = client.post("/forecast", json=payload)
    assert response.status_code == 422


def test_future_beyond_horizon_is_rejected(client: TestClient, demand_frame: pl.DataFrame) -> None:
    """学習した horizon より先を求められたら、黙って外挿せず弾くこと。"""
    payload = _payload(demand_frame, forecast_days=1)
    payload["future"][0]["date"] = str(ORIGIN + dt.timedelta(days=30))

    response = client.post("/forecast", json=payload)
    assert response.status_code == 422
    assert "horizon" in response.text


def test_negative_units_are_rejected(client: TestClient, demand_frame: pl.DataFrame) -> None:
    payload = _payload(demand_frame)
    payload["history"][0]["units_sold"] = -5

    assert client.post("/forecast", json=payload).status_code == 422


def test_empty_history_is_rejected(client: TestClient, demand_frame: pl.DataFrame) -> None:
    payload = _payload(demand_frame)
    payload["history"] = []

    assert client.post("/forecast", json=payload).status_code == 422


def test_openapi_schema_is_generated(client: TestClient) -> None:
    """OpenAPI が生成されること（クライアント自動生成の前提）。"""
    schema = client.get("/openapi.json").json()
    assert "/forecast" in schema["paths"]
