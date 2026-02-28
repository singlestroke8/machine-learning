import os
import logging
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
# 非線形モデルであるランダムフォレスト
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
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

# 前処理器の定義を関数化して、モデルのループ内で再利用できるようにする
def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            # 数値データ -> 標準化 (StandardScaler)
            ('num', StandardScaler(), numeric_features),
            # カテゴリデータ -> ワンホットエンコーディング (OneHotEncoder)
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        # 前処理器の後に残る列はそのまま通す（今回はcustomerIDなど予測に不要な列は最初から除外しているので、passthroughで問題ない）
        remainder='passthrough'
    )

def evaluate_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, model_name: str) -> dict:
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
    
    os.makedirs("models", exist_ok=True)

    try:
        logging.info(f"中間データの読み込み: {input_file}")
        df = pd.read_csv(input_file)

        # 特徴量(X)と目的変数(y)に分割
        target_col = 'Churn'
        X = df.drop(columns=[target_col, 'customerID'])
        y = df[target_col].map({'Yes': 1, 'No': 0})

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

        # パイプライン全体を学習（内部で前処理のFitとモデルのFitが順番に実行される）
        logging.info("パイプライン（前処理＋モデル）の学習を開始します...")

        # 前処理器を先に作成しておいて、モデルのループ内で再利用できるようにする
        preprocessor = build_preprocessor(numeric_features, categorical_features)

        # 3つのモデルを辞書に詰めてforループで回す
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            # class_weight='balanced' を付けることで、解約者の見落としペナルティを重くする
            "Random Forest (Balanced)": RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
            "LightGBM (Balanced)": LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
        }

        # ループの外でベストスコアを保持する変数を準備
        best_auc = 0          # 最高AUCスコア
        best_model_name = ""  # 最高モデルの名前
        best_pipeline = None  # 最高モデルのパイプライン

        logging.info("各モデルのクロスバリデーション（CV）評価を開始します。\n" + "="*45)

        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            # 【修正】X_train の中で5分割して評価（データリークなし）
            metrics = evaluate_cv(pipeline, X_train, y_train, name)
            
            # 最良モデルの判定（AUCを基準）
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_model_name = name
                best_pipeline = pipeline

        logging.info("="*45)
        logging.info(f"🏆 チャンピオンモデル: {best_model_name} (CV AUC: {best_auc:.4f})")
        
        # --- 最終評価と保存 ---
        logging.info("\nチャンピオンモデルを全学習データで再学習し、未知のテストデータで最終評価します...")
        best_pipeline.fit(X_train, y_train)
        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        
        final_auc = roc_auc_score(y_test, y_proba)
        final_rec = recall_score(y_test, y_pred)
        
        logging.info(f"✅ 最終テストデータでの AUC: {final_auc:.4f}")
        logging.info(f"✅ 最終テストデータでの Recall: {final_rec:.4f} (解約者の発見率)")
        
        joblib.dump(best_pipeline, "models/best_model_pipeline.pkl")
        logging.info("\n最良モデルを保存しました: models/best_model_pipeline.pkl")

    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()