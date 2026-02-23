import os
import logging
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def build_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """前処理とモデルを結合したパイプラインを構築する"""
    
    # 1. 前処理器の定義 (Day 8と同じロジック)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # 2. パイプラインの定義 (前処理 -> ロジスティック回帰)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    return pipeline

def main() -> None:
    input_file = "data/interim/features.csv"
    output_model_path = "models/baseline_pipeline.pkl"
    
    os.makedirs("models", exist_ok=True)

    try:
        logging.info(f"中間データの読み込み: {input_file}")
        df = pd.read_csv(input_file)

        # 特徴量(X)と目的変数(y)に分割
        target_col = 'Churn'
        X = df.drop(columns=[target_col, 'customerID'])
        y = df[target_col].map({'Yes': 1, 'No': 0})

        # 学習データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logging.info(f"データ分割完了: Train({X_train.shape[0]}行), Test({X_test.shape[0]}行)")

        # カラムの型を自動判別
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

        # パイプラインの構築
        pipeline = build_pipeline(numeric_features, categorical_features)

        # パイプライン全体を学習（内部で前処理のFitとモデルのFitが順番に実行される）
        logging.info("パイプライン（前処理＋モデル）の学習を開始します...")
        pipeline.fit(X_train, y_train)

        # テストデータで予測（内部で前処理のTransformとモデルのPredictが順番に実行される）
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] # AUC算出用（クラス1の確率）

        # 評価指標の算出
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        logging.info("=== ベースラインモデル（Logistic Regression）評価結果 ===")
        logging.info(f"Accuracy (正解率): {acc:.4f}")
        logging.info(f"Precision(適合率): {prec:.4f}")
        logging.info(f"Recall   (再現率): {rec:.4f}")
        logging.info(f"F1-score (F1値) : {f1:.4f}")
        logging.info(f"ROC-AUC  (AUC値) : {auc:.4f}")
        logging.info("==================================================")

        # パイプラインごと保存（本番環境のAPIではこれを読み込むだけで済む）
        joblib.dump(pipeline, output_model_path)
        logging.info(f"学習済みパイプラインを保存しました: {output_model_path}")

    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()