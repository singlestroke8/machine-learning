"""生成した取引明細を分析し、`docs/data-understanding.md` の数字を再現する。

    uv run python scripts/explore_transactions.py

ドキュメントに数字を手で書くと、生成器を変えたときに必ず古くなる。
主張の根拠はここで再現できるようにしておく。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

RAW_DIR = Path("data/raw")
CHURN_WINDOW_DAYS = 120


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _wape(actual: pl.Series, predicted: pl.Series) -> float:
    return float((actual - predicted).abs().sum()) / float(actual.sum())


def aggregation_levels(df: pl.DataFrame) -> None:
    """集計レベルを上げるほど読みやすくなることを示す。

    営業担当は「その時々の状況次第」と言い、経営層は「予測できるはず」と言う。
    どちらも正しく、見ている粒度が違うだけである、ということを数字で示す。
    """
    _section("1. 集計レベルと予測のしやすさ（直近5営業日の平均で予測）")
    levels: list[tuple[str, list[str]]] = [
        ("担当者 × 品名", ["営業担当者", "品名"]),
        ("部署 × 品名", ["部署", "品名"]),
        ("担当者別", ["営業担当者"]),
        ("部署別", ["部署"]),
        ("全社", []),
    ]
    for label, keys in levels:
        grouped = (
            df.group_by([*keys, "受注日"]).agg(pl.col("販売金額").sum()).sort([*keys, "受注日"])
        )
        shifted = pl.col("販売金額").shift(1).rolling_mean(5)
        grouped = grouped.with_columns(
            (shifted.over(keys) if keys else shifted).alias("予測")
        ).drop_nulls("予測")
        n_series = grouped.select(keys).unique().height if keys else 1
        print(
            f"  {label:16s} {n_series:4d} 系列   "
            f"外し率 {_wape(grouped.get_column('販売金額'), grouped.get_column('予測')):.1%}"
        )


def monthly_baselines(df: pl.DataFrame) -> None:
    """月次の売上を、素朴な方法で予測したときの外し率。"""
    _section("2. 月次・全社の素朴な予測（Excel でできる範囲）")
    monthly = (
        df.group_by(pl.col("受注日").dt.truncate("1mo").alias("月"))
        .agg(pl.col("販売金額").sum().alias("売上"))
        .sort("月")
    )
    print(f"  月次データ点数: {monthly.height}")

    recent = monthly.with_columns(
        pl.col("売上").shift(1).rolling_mean(3).alias("予測")
    ).drop_nulls()
    print(f"  直近3ヶ月の平均      : {_wape(recent['売上'], recent['予測']):.1%}")

    yoy = monthly.with_columns(pl.col("売上").shift(12).alias("前年")).drop_nulls()
    if yoy.height:
        print(f"  前年同月と同じ        : {_wape(yoy['売上'], yoy['前年']):.1%}")
        growth = float(yoy["売上"].sum()) / float(yoy["前年"].sum())
        print(
            f"  前年同月 × 成長率{growth:.3f} : "
            f"{_wape(yoy['売上'], yoy['前年'] * growth):.1%}   （{yoy.height}ヶ月で評価）"
        )


def churn_signals(df: pl.DataFrame) -> None:
    """離反の予兆が、どのくらいの強さで現れているか。"""
    from sales_analytics.data.masters import PRODUCTS, max_discount_for

    _section("3. 離反の予兆（同じ顧客の中で、末期とそれ以前を比べる）")
    positive = df.filter(pl.col("数量") > 0)
    end = positive.get_column("受注日").max()
    last = positive.group_by("顧客コード").agg(pl.col("受注日").max().alias("最終"))
    churned = last.filter(pl.col("最終") < pl.lit(end) - pl.duration(days=CHURN_WINDOW_DAYS))
    print(f"  離反顧客: {churned.height} / {last.height} 社 = {churned.height / last.height:.1%}")

    ceilings = {p.name: max_discount_for(p) for p in PRODUCTS}
    target = (
        positive.join(churned, on="顧客コード", how="inner")
        .with_columns(
            (pl.col("受注日") > pl.col("最終") - pl.duration(days=90)).alias("末期"),
            pl.col("品名").replace_strict(ceilings).alias("上限"),
        )
        .with_columns((pl.col("値引率") / pl.col("上限")).alias("交渉圧"))
    )
    print(
        target.group_by("末期")
        .agg(
            pl.len().alias("明細数"),
            pl.col("交渉圧").mean().round(4).alias("値引き上限に対する比"),
            pl.col("数量").mean().round(1).alias("平均数量"),
        )
        .sort("末期")
    )
    print("  → 「値引率が上がる」「1回あたりが小さくなる」が同時に、弱く現れる")


def cross_sell(df: pl.DataFrame) -> None:
    """同時購入の偏り。推薦の課題が成立するかどうかの目安。"""
    _section("4. 商材の関連購買（推薦の課題が成立するか）")
    pairs = (
        df.filter(pl.col("数量") > 0)
        .group_by("受注番号")
        .agg(pl.col("品名").sort().alias("組"))
        .filter(pl.col("組").list.len() == 2)
        .with_columns(pl.col("組").list.join(" + ").alias("組合せ"))
    )
    counts = pairs.get_column("組合せ").value_counts().sort("count", descending=True)
    top, bottom = counts.get_column("count")[0], counts.get_column("count")[-1]
    print(f"  2品目同時購入 {pairs.height:,} 件 / 組合せ {counts.height} 種")
    print(f"  最頻 {top} 回 vs 最少 {bottom} 回 = {top / bottom:.0f} 倍の偏り")
    print(f"  最頻の組合せ: {counts.get_column('組合せ')[0]}")


def anomalies(df: pl.DataFrame, labels: pl.DataFrame) -> None:
    """異常の混入と、素朴な検知が誤検知する量。"""
    _section("5. 異常検知の題材（素朴な検知はどれだけ外すか）")
    print(labels.get_column("異常種別").value_counts().sort("count", descending=True))
    negative = df.filter(pl.col("数量") < 0)
    flagged = negative.join(
        labels.select("受注番号", "明細番号").unique(), on=["受注番号", "明細番号"], how="semi"
    )
    rate = 1 - flagged.height / negative.height
    print(f"  数量が負の行 {negative.height} 行のうち、異常は {flagged.height} 行")
    print(f"  → 「数量が負なら異常」という検知は {rate:.0%} が誤検知になる")


def main() -> None:
    transactions_path = RAW_DIR / "transactions.csv"
    if not transactions_path.exists():
        print(f"{transactions_path} がありません。")
        print("先に `uv run sales generate` を実行してください。")
        raise SystemExit(1)

    df = pl.read_csv(transactions_path, try_parse_dates=True)
    labels = pl.read_csv(RAW_DIR / "anomaly_labels.csv")
    print(f"読み込み: {transactions_path}（{df.height:,} 行 × {df.width} 列）")

    aggregation_levels(df)
    monthly_baselines(df)
    churn_signals(df)
    cross_sell(df)
    anomalies(df, labels)


if __name__ == "__main__":
    main()
