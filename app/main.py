from fastapi import FastAPI

# 1. FastAPIインスタンスの作成（APIアプリケーション本体）
app = FastAPI(
    title="Churn Prediction API",
    description="顧客の解約(Churn)を予測する機械学習API",
    version="1.0.0"
)

# 2. ルートパス（一番大元のURL）へのアクセス時の動作
@app.get("/")
def read_root():
    """APIのトップページにアクセスした際の挨拶メッセージを返します"""
    return {"message": "Welcome to the Churn Prediction API!"}

# 3. ヘルスチェック用のエンドポイント（死活監視用）
@app.get("/health")
def health_check():
    """システム（ロードバランサー等）がAPIの稼働状況を確認するためのエンドポイント"""
    return {
        "status": "ok",
        "message": "API is running correctly."
    }