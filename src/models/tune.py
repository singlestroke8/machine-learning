import json
import logging
import os

import joblib
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optunaのログが多すぎるので、INFOレベル（最適化の結果のみ）に制限
optuna.logging.set_verbosity(optuna.logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """前処理器の構築（Day 8から共通のモジュール）"""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

def save_metrics(metrics: dict, output_path: str) -> None:
    """評価指標をJSONファイルとして保存する"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"評価指標を保存しました: {output_path}")

def main() -> None:
    input_file = "data/interim/features.csv"
    os.makedirs("models", exist_ok=True)

    try:
        logging.info("データの読み込みと分割を開始します...")
        df = pd.read_csv(input_file)
        target_col = 'Churn'
        X = df.drop(columns=[target_col, 'customerID'])
        y = df[target_col].map({'Yes': 1, 'No': 0})

        # テストデータ(20%)は完全に切り離し、チューニングは学習データ(80%)内でのCVで行う
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
        preprocessor = build_preprocessor(numeric_features, categorical_features)

        # Optunaの目的関数（Objective Function）の定義
        def objective(trial):
            # 1. 探索するハイパーパラメータの範囲を定義
            params = {
                'random_state': 42,
                'class_weight': 'balanced', # 必須: 不均衡データ対策
                'verbose': -1,              # LightGBMの不要な警告を消す
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }

            # 2. 提案されたパラメータでモデルとパイプラインを作成
            model = LGBMClassifier(**params)
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            # 3. 5-Fold CVで「F1-score」を計算
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            f1_scorer = make_scorer(f1_score)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=f1_scorer, n_jobs=-1)
            
            # 5回のテストの平均F1スコアを返す（これを最大化するようにOptunaが動く）
            return scores.mean()

        logging.info("Optunaによるハイパーパラメータの探索を開始します（20回のトライアル）...\n" + "="*50)
        
        # スタディの作成（F1スコアを「最大化(maximize)」する方向で探索）
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        
        # 探索の実行（手元のPCでも数分で終わるように n_trials=20 に設定）
        study.optimize(objective, n_trials=20)

        logging.info("="*50)
        logging.info("🏆 チューニング完了！")
        logging.info(f"ベストF1スコア (CV平均): {study.best_value:.4f}")
        logging.info("ベストパラメータ:")
        for key, value in study.best_params.items():
            logging.info(f"  {key}: {value}")

        # --- チューニング後のベストパラメータで最終モデルを作成 ---
        logging.info("\nベストパラメータを使って全学習データで最終モデルを構築・評価します...")
        
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['class_weight'] = 'balanced'
        best_params['verbose'] = -1

        best_model = LGBMClassifier(**best_params)
        best_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', best_model)
        ])

        # 学習データ全体でFit
        best_pipeline.fit(X_train, y_train)
        
        # 未知のテストデータで最終評価
        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]

        # 計算結果を一度「変数」に格納する
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        logging.info("=== 最終モデル (Tuned LightGBM) テストデータ評価 ===")
        logging.info(f"Accuracy : {acc:.4f}")
        logging.info(f"Precision: {prec:.4f}")
        logging.info(f"Recall   : {rec:.4f} (解約者の発見率)")
        logging.info(f"F1-score : {f1:.4f}")
        logging.info(f"ROC-AUC  : {auc:.4f}")
        
        # 指標を辞書（キーと値のペア）にまとめる
        final_metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4)
        }
        
        # reports フォルダに JSON として保存
        save_metrics(final_metrics, "reports/metrics.json")

        # モデルの保存
        model_path = "models/tuned_lightgbm_pipeline.pkl"
        joblib.dump(best_pipeline, model_path)
        logging.info(f"\n✅ チューニング済みモデルを保存しました: {model_path}")

    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()