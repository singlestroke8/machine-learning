import logging
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main() -> None:
    model_path = "models/tuned_lightgbm_pipeline.pkl"
    output_dir = "reports/figures"
    
    # 画像を保存するディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    try:
        logging.info(f"学習済みモデルを読み込んでいます: {model_path}")
        pipeline = joblib.load(model_path)

        # 1. パイプラインから「前処理器」と「分類器(LightGBM)」を別々に取り出す
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']

        # 2. 特徴量名（列名）の復元
        # OneHotEncoder等で変換・増殖した後の正確な列名を取得する（超重要スキル）
        feature_names = preprocessor.get_feature_names_out()
        
        # 見栄えを良くするため、前処理器が勝手につける接頭辞('num__', 'cat__')を削除
        clean_feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

        # 3. モデルから特徴量重要度（点数）を取り出す
        importances = classifier.feature_importances_

        # 4. 列名と点数をPandasデータフレームにまとめ、点数が高い順に並び替え
        importance_df = pd.DataFrame({
            'Feature': clean_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # 5. 可視化（Seabornを使って美しい横棒グラフを作成）
        logging.info("特徴量重要度のグラフを描画しています...")
        plt.figure(figsize=(12, 8))
        
        # 上位20個の特徴量のみを描画
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=importance_df.head(20), 
            hue='Feature',       # 色分けのため
            palette='viridis',   # プロフェッショナルな配色
            legend=False
        )
        
        plt.title('Top 20 Feature Importances (Tuned LightGBM)', fontsize=16)
        plt.xlabel('Importance Score (Number of splits)', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()

        # 6. グラフをPNG画像として保存
        output_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(output_path, dpi=300) # 高解像度で保存
        logging.info(f"\n グラフを画像として保存しました: {output_path}")

        # ターミナルにもトップ10をテキストで表示
        logging.info("\n=== 👑 トップ10の重要な特徴量 ===")
        for index, row in importance_df.head(10).iterrows():
            logging.info(f"{row['Feature']:<35}: {row['Importance']:.1f}")
        logging.info("=================================")

    except FileNotFoundError:
        logging.error("エラー: モデルファイルが見つかりません。先に train.py や tune.py を実行してください。")
    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()