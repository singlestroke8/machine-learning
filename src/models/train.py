import os
import sys
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# utilsをインポート
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_data, build_preprocessor, save_model, save_metrics

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def evaluate_cv(pipeline: Pipeline, X, y, model_name: str) -> dict:
    """【修正】5-Fold クロスバリデーションによる厳密な評価を行う"""
    # ターゲットの割合（解約の有無）を維持したまま5分割する設定
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 計算したい評価指標のリスト
    scoring = {'acc': 'accuracy', 'prec': 'precision', 'rec': 'recall', 'f1': 'f1', 'auc': 'roc_auc'}
    
    # CVの実行（n_jobs=-1 でPCの全コアを使って並列計算）
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    # 5回のテスト結果の「平均値」を算出
    mean_acc = cv_results['test_acc'].mean()
    mean_prec = cv_results['test_prec'].mean()
    mean_rec = cv_results['test_rec'].mean()
    mean_f1 = cv_results['test_f1'].mean()
    mean_auc = cv_results['test_auc'].mean()
    
    logging.info(f"--- {model_name} (5-Fold CV 平均) ---")
    logging.info(f"Accuracy : {mean_acc:.4f}")
    logging.info(f"Precision: {mean_prec:.4f}")
    logging.info(f"Recall   : {mean_rec:.4f}")
    logging.info(f"F1-score : {mean_f1:.4f}")
    logging.info(f"ROC-AUC  : {mean_auc:.4f}\n")
    
    return {"acc": mean_acc, "prec": mean_prec, "rec": mean_rec, "f1": mean_f1, "auc": mean_auc}

# 評価とログ出しを行う専用の関数
def evaluate_and_log(y_true, y_pred, y_proba, model_name: str) -> dict:
    """評価指標を計算し、ログに出力する"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    logging.info(f"--- {model_name} 評価結果 ---")
    logging.info(f"Accuracy (正解率): {acc:.4f}")
    logging.info(f"Precision(適合率): {prec:.4f}")
    logging.info(f"Recall   (再現率): {rec:.4f}")
    logging.info(f"F1-score (F1値) : {f1:.4f}")
    logging.info(f"ROC-AUC  (AUC値) : {auc:.4f}\n")
    
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

def main() -> None:
    input_file = "data/interim/features.csv"

    try:
        # utilsを使ってデータを読み込む
        X, y = load_data(input_file)

        # 最終確認用のテストデータ（20%）を切り離す。CVは残りの80%（X_train）で行う。
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logging.info(f"データ分割完了: Train({X_train.shape[0]}行), Test({X_test.shape[0]}行)")

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

        logging.info("\n--- クロスバリデーション評価を開始 ---")
        metrics = evaluate_cv(pipeline, X_train, y_train, "LightGBM (Baseline)")

        # 最終学習と評価
        logging.info("\n--- 最終モデルの学習と評価 ---")
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
        
        logging.info(f"最終テストデータ Recall: {final_metrics['recall']}")

        # 3. utilsを使って保存する（2行でスッキリ！）
        save_metrics(final_metrics, "reports/train_metrics.json")
        save_model(pipeline, "models/lightgbm_pipeline.pkl")

    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()