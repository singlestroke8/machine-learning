import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def plot_categorical_vs_target(df: pd.DataFrame, col: str, target: str, output_path: str) -> None:
    """カテゴリ変数とターゲット変数の関係を棒グラフで出力する"""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=col, hue=target, palette="Set2")
    plt.title(f"{col} vs {target}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved plot: {output_path}")

def plot_numerical_vs_target(df: pd.DataFrame, col: str, target: str, output_path: str) -> None:
    """数値変数とターゲット変数の関係をヒストグラム（積み上げ）で出力する"""
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=col, hue=target, multiple="stack", palette="Set2", kde=True)
    plt.title(f"Distribution of {col} by {target}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved plot: {output_path}")

def main() -> None:
    input_file = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_dir = "reports/figures"

    # 出力先ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_file)
        logging.info(f"Data loaded successfully: {df.shape}")

        # 1. 契約期間 (tenure) と 解約 (Churn) の関係
        plot_numerical_vs_target(df, "tenure", "Churn", os.path.join(output_dir, "tenure_vs_churn.png"))

        # 2. 月額料金 (MonthlyCharges) と 解約 (Churn) の関係
        plot_numerical_vs_target(df, "MonthlyCharges", "Churn", os.path.join(output_dir, "monthly_charges_vs_churn.png"))

        # 3. 契約形態 (Contract) と 解約 (Churn) の関係
        plot_categorical_vs_target(df, "Contract", "Churn", os.path.join(output_dir, "contract_vs_churn.png"))

        logging.info("すべてのグラフの出力が完了しました。")

    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    main()