import os
import sys
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

# srcディレクトリ内のモジュールを読み込めるようにパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils import load_data, build_preprocessor

def test_load_data(tmp_path) -> None:
    """load_data関数が正しくデータを分割・変換できるかをテストする"""
    
    # 1. テスト用の「ダミーデータ」を一時ファイルとして作成
    # (tmp_pathはpytestが自動で用意・削除してくれる安全な一時フォルダです)
    dummy_csv = tmp_path / "dummy_data.csv"
    dummy_df = pd.DataFrame({
        "customerID": ["001-A", "002-B", "003-C"],
        "tenure": [1, 10, 20],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "Churn": ["Yes", "No", "Yes"]
    })
    dummy_df.to_csv(dummy_csv, index=False)

    # 2. テスト対象の関数を実行
    X, y = load_data(str(dummy_csv))

    # 3. 検証 (assert文を使って「こうなっているはずだ！」を確認する)
    # IDや目的変数が特徴量(X)から消えているか？
    assert "customerID" not in X.columns, "customerIDがXから削除されていません"
    assert "Churn" not in X.columns, "ChurnがXから削除されていません"
    
    # Yes/No が 1/0 に正しく変換されているか？
    assert list(y) == [1, 0, 1], "Churnの Yes/No が 1/0 に正しく変換されていません"
    
    # 行数が変わっていないか？
    assert len(X) == 3, "読み込みの前後で行数が変わってしまっています"


def test_build_preprocessor() -> None:
    """build_preprocessor関数が正しいオブジェクトを返すかをテストする"""
    numeric_features = ["tenure"]
    categorical_features = ["InternetService"]

    # 関数を実行
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # ColumnTransformerというScikit-learnの部品が正しく作られているか？
    assert isinstance(preprocessor, ColumnTransformer), "返り値がColumnTransformerではありません"