import os
import sys
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as TestClient

# （appフォルダがある場所）をPythonに教える
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app

# 1. テスト用のクライアントを準備（サーバーを立ち上げずに内部で通信テストができる魔法のツール）
@pytest.fixture
def client():
    # withブロックを使うことで、main.pyで定義した「lifespan（モデルの読み込み）」がテスト時にも自動で実行されます
    with TestClient(app) as c:
        yield c

# --- 2. ヘルスチェックのテスト ---
def test_health_check(client):
    """ヘルスチェックエンドポイントが正常に動作するかをテスト"""
    response = client.get("/health")
    
    # HTTPステータスコードが200 (OK) であること
    assert response.status_code == 200
    # 返ってくるJSONの中身が期待通りであること
    assert response.json() == {"status": "ok", "message": "API is running correctly."}

# --- 3. 正常系（正しいデータ）の推論テスト ---
def test_predict_success(client):
    """正しい顧客データを送信した場合、AIが推論結果を返すかをテスト（正常系）"""
    # Swagger UIで確認した時と同じ、完全なダミーデータ（20項目）
    valid_payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "MonthlyCharges": 85.50,
        "TotalCharges": 1026.00,
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Num_Additional_Services": 2
    }
    
    # POSTリクエストを送信
    response = client.post("/predict", json=valid_payload)
    
    # 推論成功なら200が返るはず
    assert response.status_code == 200
    
    data = response.json()
    # レスポンスに 'prediction' (0か1) と 'probability' (0.0~1.0の小数) が含まれているか検証
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert "message" in data

# --- 4. 異常系（間違ったデータ）のバリデーションテスト ---
def test_predict_validation_error(client):
    """不正なデータを送信した場合、Pydanticが弾いてエラーを返すかをテスト（異常系）"""
    # わざと必須項目を削り、tenure（数値）に文字列を入れた「ダメなデータ」
    invalid_payload = {
        "gender": "Female",
        "tenure": "十二ヶ月"  # エラーの元！
    }
    
    response = client.post("/predict", json=invalid_payload)
    
    # FastAPI(Pydantic)が自動で弾いて 422 Unprocessable Entity を返すはず！
    assert response.status_code == 422