"""取引明細を分析し、`docs/data-understanding.md` の数字を再現する。

    uv run python scripts/explore_transactions.py

ドキュメントに数字を手で書くと、生成器を変えたときに必ず古くなる。
主張の根拠はここで再現できるようにしておく。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

DEFAULT_PATH = Path("data/raw/transactions.csv")


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def analyze(df: pl.DataFrame) -> None:
    """ドキュメントに載せた主張を、上から順に検算する。"""
    _section("1. 月次の季節性（MONTH_FACTOR に対応）")
    monthly = (
        df.group_by(pl.col("受注日").dt.month().alias("月"))
        .agg(pl.col("販売金額").sum().alias("売上"))
        .sort("月")
    )
    total = monthly.get_column("売上").sum()
    print(monthly.with_columns((pl.col("売上") / total * 12).round(3).alias("指数")))

    _section("2. 曜日（WEEKDAY_FACTOR に対応）")
    print(
        df.group_by(pl.col("受注日").dt.weekday().alias("曜日"))
        .agg(pl.col("販売金額").sum().alias("売上"), pl.len().alias("明細数"))
        .sort("曜日")
    )

    _section("3. 部署×カテゴリ（DEPARTMENT_AFFINITY に対応）")
    print(
        df.group_by("部署", "品名カテゴリ")
        .agg(pl.col("販売金額").sum().alias("売上"))
        .pivot(on="品名カテゴリ", index="部署", values="売上")
    )

    _section("4. 年次トレンド（yearly_growth に対応）")
    print(
        df.group_by(pl.col("受注日").dt.year().alias("年"))
        .agg(pl.col("販売金額").sum().alias("売上"), pl.col("粗利").sum().alias("粗利"))
        .sort("年")
        .with_columns((pl.col("粗利") / pl.col("売上") * 100).round(2).alias("粗利率%"))
    )

    _section("5. 担当者別の売上と粗利率（定数には無い、二次的な結果）")
    print(
        df.group_by("営業担当者", "部署")
        .agg(
            pl.col("販売金額").sum().alias("売上"),
            (pl.col("粗利").sum() / pl.col("販売金額").sum() * 100).round(2).alias("粗利率%"),
        )
        .sort("売上", descending=True)
    )

    _section("6. 値引率と数量の交絡（設計していない結果）")
    centered = df.with_columns(
        (pl.col("値引率") - pl.col("値引率").mean().over(["顧客コード", "品名"])).alias("値引_c"),
        (pl.col("数量") - pl.col("数量").mean().over(["顧客コード", "品名"])).alias("数量_c"),
    )
    raw_corr = df.select(pl.corr("値引率", "数量")).item()
    within_corr = centered.select(pl.corr("値引_c", "数量_c")).item()
    by_customer = df.group_by("顧客コード").agg(
        pl.col("販売金額").sum().alias("総売上"), pl.col("値引率").mean().alias("平均値引率")
    )
    size_corr = by_customer.select(pl.corr("総売上", "平均値引率")).item()
    print(f"  値引率 × 数量（そのまま）          : {raw_corr:.3f}")
    print(f"  値引率 × 数量（顧客×品名で中心化） : {within_corr:.3f}")
    print(f"  顧客の総売上 × 平均値引率          : {size_corr:.3f}")
    print("  → そのままの相関の大半は、顧客規模による交絡である")

    _section("7. 実データにあって、ここに無いもの")
    spans = df.group_by("顧客コード").agg(
        pl.col("受注日").min().alias("初回"), pl.col("受注日").max().alias("最終")
    )
    # 「初回受注日が期間の先頭」かどうかで数えると、たまたま初日に発注が無かった
    # だけの顧客を新規と誤認する。1ヶ月の余裕を見て、実質的な出入りだけを数える。
    margin = pl.duration(days=31)
    start = df.get_column("受注日").min()
    end = df.get_column("受注日").max()
    late = spans.filter(pl.col("初回") > start + margin).height
    early = spans.filter(pl.col("最終") < end - margin).height
    price_changes = (
        df.group_by("品名").agg(pl.col("定価").n_unique().alias("定価の種類")).sort("品名")
    )
    print(f"  期間開始の1ヶ月後より後に初受注した顧客（新規獲得）: {late} 社 / {spans.height} 社")
    print(f"  期間終了の1ヶ月前より前で受注が途切れた顧客（離脱）: {early} 社 / {spans.height} 社")
    print(
        f"  定価が2種類以上ある品名   : {price_changes.filter(pl.col('定価の種類') > 1).height} 件"
    )
    print(f"  数量が負の行（返品・取消）: {df.filter(pl.col('数量') < 0).height} 行")


def main() -> None:
    if not DEFAULT_PATH.exists():
        print(f"{DEFAULT_PATH} がありません。")
        print("先に `uv run dfc generate-transactions` を実行してください。")
        raise SystemExit(1)
    df = pl.read_csv(DEFAULT_PATH, try_parse_dates=True)
    print(f"読み込み: {DEFAULT_PATH}（{df.height:,} 行 × {df.width} 列）")
    analyze(df)


if __name__ == "__main__":
    main()
