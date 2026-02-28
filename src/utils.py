import os
import json
import logging
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- 1. データ読み込みの共通化 ---
def load_data(file_path: str, target_col: str = 'Churn') -> tuple:
    """データを読み込み、特徴量(X)と正解ラベル(y)に分割する"""
    logging.info(f"データを読み込んでいます: {file_path}")
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col, 'customerID'], errors='ignore')
    # Yes/No を 1/0 に変換
    y = df[target_col].map({'Yes': 1, 'No': 0}) if df[target_col].dtype == 'object' else df[target_col]
    return X, y

# --- 2. 前処理定義の共通化 ---
def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """数値・カテゴリそれぞれに対する前処理器を構築する"""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

# --- 3. 保存処理の共通化 ---
def save_model(model, output_path: str) -> None:
    """学習済みモデルを保存する"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    logging.info(f"✅ モデルを保存しました: {output_path}")

def save_metrics(metrics: dict, output_path: str) -> None:
    """評価指標をJSONファイルとして保存する"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"✅ 評価指標を保存しました: {output_path}")