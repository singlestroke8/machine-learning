import os
import sys
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# utilsから setup_logger もインポートする
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_data, build_preprocessor, save_model, save_metrics, setup_logger

# 簡易設定（logging.basicConfig）を削除し、共通ロガーを呼び出す
logger = setup_logger(__name__)

def evaluate_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, float]:
    # ターゲットの割合（解約の有無）を維持したまま5分割する設定
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 計算したい評価指標のリスト
    scoring = {'acc': 'accuracy', 'prec': 'precision', 'rec': 'recall', 'f1': 'f1', 'auc': 'roc_auc'}
    
    # CVの実行（n_jobs=-1 でPCの全コアを使って並列計算）
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    # 5回のテスト結果の「平均値」を算出
    metrics: Dict[str, float] = {
        "acc": float(cv_results['test_acc'].mean()),
        "prec": float(cv_results['test_prec'].mean()),
        "rec": float(cv_results['test_rec'].mean()),
        "f1": float(cv_results['test_f1'].mean()),
        "auc": float(cv_results['test_auc'].mean())
    }
    logger.info(f"[{model_name}] CV F1-score: {metrics['f1']:.4f} / Recall: {metrics['rec']:.4f}")
    return metrics

def main() -> None:
    input_file = "data/interim/features.csv"

    try:
        # utilsを使ってデータを読み込む
        X, y = load_data(input_file)

        # 最終確認用のテストデータ（20%）を切り離す。CVは残りの80%（X_train）で行う。
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # カラムの型を自動判別
        # 数値カラムとカテゴリカルカラムをリスト化して、前処理器の定義に利用する
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # 文字列カラムをカテゴリカルカラムとみなす（このデータセットではobject型がカテゴリカル）
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

        # utilsを使って前処理器を作る
        preprocessor = build_preprocessor(numeric_features, categorical_features)

        # モデルの定義とパイプライン構築
        model = LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

        logger.info("\n--- クロスバリデーション評価を開始 ---")
        # metrics = evaluate_cv(pipeline, X_train, y_train, "LightGBM (Baseline)")

        # 最終学習と評価
        logger.info("\n--- 最終モデルの学習と評価 ---")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        final_metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4)
        }
        
        logger.info(f"最終テストデータ Recall: {final_metrics['recall']}")

        # 3. utilsを使って保存する（2行でスッキリ！）
        save_metrics(final_metrics, "reports/train_metrics.json")
        save_model(pipeline, "models/lightgbm_pipeline.pkl")

    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()