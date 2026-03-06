from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import logging

# 先ほど作成したスキーマ（型定義）をインポート
from app.schemas import ChurnRequest, ChurnResponse

logger = logging.getLogger("uvicorn.error")

# 学習済みモデルをメモリに保持するためのグローバルな「箱」
ml_models = {}

# --- 1. サーバー起動時・終了時の処理（ライフサイクル管理） ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """サーバー起動時に1回だけモデルを読み込み、終了時に解放する"""
    # 読み込むモデルのパス（Week 3で保存したパイプライン）
    model_path = "models/lightgbm_pipeline.pkl" 
    
    try:
        # モデルをロードして辞書に格納
        ml_models["churn_model"] = joblib.load(model_path)
        logger.info("✅ 学習済みモデルの読み込みに成功しました。")
    except Exception as e:
        logger.error(f"❌ モデルの読み込みに失敗しました: {e}")
        ml_models["churn_model"] = None
        
    yield # ここでサーバーが稼働し、リクエストを待ち受けます
    
    # サーバー終了時の処理
    ml_models.clear()
    logger.info("サーバーを停止し、メモリを解放しました。")

# --- 2. FastAPIインスタンスの作成 ---
app = FastAPI(
    title="Churn Prediction API",
    description="顧客の解約(Churn)を予測する機械学習API",
    version="1.0.0",
    lifespan=lifespan  # 先ほどの起動処理をセット
)

# --- 3. 既存のヘルスチェック ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API!"}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running correctly."}

# --- 4. 【新規】推論エンドポイント ---
# response_modelを指定することで、出力もPydanticが型チェックしてくれます
@app.post("/predict", response_model=ChurnResponse)
def predict_churn(request: ChurnRequest):
    """
    顧客データを受け取り、解約確率と予測クラスを返す
    """
    # モデルが読み込めていない場合は503エラーを返す
    if ml_models.get("churn_model") is None:
        raise HTTPException(status_code=503, detail="モデルが準備できていません。")

    try:
        # 1. Pydanticモデル(JSON)を辞書に変換し、PandasのDataFrameにする
        try:
            # Pydantic v2 の場合
            input_data = request.model_dump()
        except AttributeError:
            # Pydantic v1 の場合（念のためのフォールバック）
            input_data = request.dict()
            
        input_df = pd.DataFrame([input_data])

        # 2. メモリ上のモデルを使って推論を実行
        model = ml_models["churn_model"]
        # .predict() で 0 か 1 のクラスを予測
        prediction = int(model.predict(input_df)[0])
        # .predict_proba() で クラス1（解約）の確率を取得
        probability = float(model.predict_proba(input_df)[0][1])

        # 3. ビジネス向けの補足メッセージを生成
        if prediction == 1:
            message = "High risk of churn (解約リスク高)"
        else:
            message = "Low risk of churn (解約リスク低)"

        # 4. Pydanticのレスポンススキーマに当てはめて返す
        return ChurnResponse(
            prediction=prediction,
            probability=probability,
            message=message
        )

    except Exception as e:
        logger.error(f"推論中にエラーが発生しました: {e}")
        # 万が一計算エラーが起きても、サーバーを落とさずに500エラーを返す
        raise HTTPException(status_code=500, detail="サーバー内部で推論エラーが発生しました。")