import os
import logging
import pandas as pd
import numpy as np

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """既存のデータから新しい特徴量を生成・クリーニングする"""
    df_featured = df.copy()

    # 1. 暗黙のノイズ処理: 'TotalCharges'（総支払額）の空白を処理し、数値型に変換
    # Kaggleのこのデータセット特有の罠。新規顧客(tenure=0)の支払額が空白スペースになっている
    df_featured['TotalCharges'] = pd.to_numeric(df_featured['TotalCharges'].replace(" ", np.nan))
    
    # 契約期間0ヶ月の顧客の総支払額を0で補完
    df_featured['TotalCharges'] = df_featured['TotalCharges'].fillna(0)

    # 2. 新規特徴量の作成: 付加サービスの契約数（ロックイン効果の測定）
    # 複数のサービス列において 'Yes' となっている数を横方向にカウントする
    services = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df_featured['Num_Additional_Services'] = df_featured[services].apply(lambda x: (x == 'Yes').sum(), axis=1)

    return df_featured

def main() -> None:
    input_file = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_dir = "data/interim"
    output_file = os.path.join(output_dir, "features.csv")

    # 出力先ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    try:
        logging.info(f"生データの読み込み: {input_file}")
        df = pd.read_csv(input_file)

        logging.info("特徴量エンジニアリングを実行中...")
        df_featured = create_features(df)

        # 処理結果を中間データとして保存
        df_featured.to_csv(output_file, index=False)
        logging.info(f"特徴量生成済みデータを保存しました: {output_file}")
        logging.info(f"データサイズ: {df_featured.shape}")

    except FileNotFoundError:
        logging.error(f"ファイルが見つかりません: {input_file}")
    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()