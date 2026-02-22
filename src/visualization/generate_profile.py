import pandas as pd
from ydata_profiling import ProfileReport
import logging
import os

# ロギングの設定（printの代わりに実務で使う標準的な方法）
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    input_file = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_file = "reports/eda_report.html"

    logging.info(f"データの読み込みを開始します: {input_file}")
    
    try:
        # データの読み込み
        df = pd.read_csv(input_file)
        logging.info(f"データサイズ: {df.shape[0]}行, {df.shape[1]}列")

        # プロファイリングレポートの作成
        # explorative=True にすることで、より詳細な分析（相関など）が行われます
        logging.info("プロファイリングレポートを生成中...（数分かかる場合があります）")
        profile = ProfileReport(df, title="Telco Customer Churn EDA Report", explorative=True)

        # レポートをHTMLファイルとして出力
        profile.to_file(output_file)
        logging.info(f"レポートの出力が完了しました: {output_file}")

    except FileNotFoundError:
        logging.error(f"ファイルが見つかりません: {input_file}")
    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()