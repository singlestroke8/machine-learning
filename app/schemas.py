from pydantic import BaseModel, Field

# 1. クライアントから受け取るデータ（入力）の設計図
class ChurnRequest(BaseModel):
    """
    推論APIが受け取る顧客データのスキーマ
    ※ここで定義した変数名が、そのままPandasの列名になります
    """
    gender: str = Field(..., example="Female", description="顧客の性別 (Male / Female)")
    SeniorCitizen: int = Field(..., example=0, description="高齢者かどうか (1: Yes, 0: No)")
    Partner: str = Field(..., example="Yes", description="パートナーがいるか (Yes / No)")
    Dependents: str = Field(..., example="No", description="扶養家族がいるか (Yes / No)")
    tenure: int = Field(..., example=12, description="契約期間（ヶ月）")
    InternetService: str = Field(..., example="Fiber optic", description="インターネットサービスの種類")
    Contract: str = Field(..., example="Month-to-month", description="契約形態")
    MonthlyCharges: float = Field(..., example=85.50, description="月額料金")
    TotalCharges: float = Field(..., example=1026.00, description="総支払額")
    
    # --- モデルが要求している不足項目を追加 ---
    PaymentMethod: str = Field(..., example="Electronic check", description="支払い方法")
    PaperlessBilling: str = Field(..., example="Yes", description="ペーパーレス決済")
    PhoneService: str = Field(..., example="Yes", description="電話サービス")
    MultipleLines: str = Field(..., example="No", description="複数回線")
    OnlineSecurity: str = Field(..., example="No", description="オンラインセキュリティ")
    OnlineBackup: str = Field(..., example="Yes", description="オンラインバックアップ")
    DeviceProtection: str = Field(..., example="No", description="デバイス保護")
    TechSupport: str = Field(..., example="No", description="テクニカルサポート")
    StreamingTV: str = Field(..., example="Yes", description="ストリーミングTV")
    StreamingMovies: str = Field(..., example="No", description="ストリーミング映画")
    
    # --- 特徴量エンジニアリングで追加した項目 ---
    Num_Additional_Services: int = Field(..., example=2, description="追加サービスの利用数")

# 2. クライアントへ返すデータ（出力）の設計図
class ChurnResponse(BaseModel):
    """
    推論APIが返す予測結果のスキーマ
    """
    prediction: int = Field(..., example=1, description="予測クラス (1: 解約する, 0: 解約しない)")
    probability: float = Field(..., example=0.8245, description="解約確率 (0.0 ~ 1.0)")
    message: str = Field(..., example="High risk of churn", description="補足メッセージ")