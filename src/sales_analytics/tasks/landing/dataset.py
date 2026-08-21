"""着地予測の学習データを組み立てる。

## 1行の意味

    （部署, 対象月, 経過営業日 k） → その月の着地額

「10月の10営業日目に立っている」という状況を1行で表す。
同じ月について、1営業日目から月末まで行が並ぶ。

## リークを構造的に防ぐ

この課題で最も起きやすい事故は、**k日目の予測に、k日目より後の受注を混ぜる**
ことである。エラーは出ず、検証の成績だけが良くなる。

そこで特徴量は、次の2種類からしか作らない。

1. **当月の k 日目までの累計**（``cum_`` で始まる列）
2. **前月以前に確定した実績**（``prev_`` で始まる列）

``cum_`` は営業日グリッド上の累積和で作るので、定義上 k 日目までしか含まない。
``prev_`` は対象月より前のデータしか参照しない。
この2つ以外の作り方をしない、というのが唯一の規約である。

検査は ``tests/test_landing_leakage.py`` に置いた。
「k日目より後の受注を10倍に改変しても、k日目の特徴量が1ビットも変わらない」
という**外から観測できる性質**で縛っている。
"""

from __future__ import annotations

import polars as pl

from sales_analytics.features.calendar import business_days
from sales_analytics.polars_utils import as_date

#: 予測の単位を決める列
GROUP_COL = "部署"
MONTH_COL = "対象月"
STEP_COL = "経過営業日"
TARGET_COL = "着地額"

#: 全社を、部署と同じ形の1系列として扱うときのラベル
COMPANY_LABEL = "全社"

#: 特徴量の尺度をそろえる基準。系列と年で桁が違う問題を消す
SCALE_COL = "prev_前年同月の着地"

#: 「残りいくら入るか」を測る物差し（円）。前年同月の着地額を使う。
#: モデルは着地額そのものではなく、**残りをこの物差しで割った比**を学習する。
#: 理由は ``train._remaining_target`` を参照。
ANCHOR_COL = SCALE_COL

#: 特徴量として使う列の接頭辞。
#: ``cum_`` と ``prev_`` は中間列であって特徴量ではない（金額のままで尺度を持つため）。
FEATURE_PREFIXES = ("cal_", "feat_")

_AMOUNT = "販売金額"
_DATE = "受注日"


def _business_day_grid(start: object, end: object) -> pl.DataFrame:
    """期間内の営業日に、月内の連番と月の営業日数を振った表を作る。

    受注のあった日だけを並べると、受注が1件も無かった営業日が抜けて
    ``k`` がずれる。暦から作った完全な営業日の列を土台にする。
    """
    days = business_days(as_date(start), as_date(end))
    frame = pl.DataFrame({_DATE: days}).with_columns(
        pl.col(_DATE).dt.truncate("1mo").alias(MONTH_COL)
    )
    return frame.with_columns(
        pl.col(_DATE).rank("ordinal").over(MONTH_COL).cast(pl.Int32).alias(STEP_COL),
        pl.len().over(MONTH_COL).cast(pl.Int32).alias("cal_月の営業日数"),
    )


def _daily_by_group(transactions: pl.DataFrame, *, include_company: bool) -> pl.DataFrame:
    """部署ごと（＋全社）の日次実績を作る。"""
    by_department = transactions.group_by(GROUP_COL, _DATE).agg(
        pl.col(_AMOUNT).sum().alias("金額"),
        pl.col("受注番号").n_unique().alias("受注件数"),
        pl.col("顧客コード").n_unique().alias("顧客数"),
    )
    if not include_company:
        return by_department
    company = (
        transactions.group_by(_DATE)
        .agg(
            pl.col(_AMOUNT).sum().alias("金額"),
            pl.col("受注番号").n_unique().alias("受注件数"),
            pl.col("顧客コード").n_unique().alias("顧客数"),
        )
        .with_columns(pl.lit(COMPANY_LABEL).alias(GROUP_COL))
    )
    return pl.concat([by_department, company.select(by_department.columns)])


def _add_cumulative(panel: pl.DataFrame) -> pl.DataFrame:
    """当月の k 日目までの累計を作る。

    ``cum_`` の列は、営業日順に並べた累積和で定義する。
    この作り方をしている限り、k 日目より後の受注は構造的に入らない。
    """
    keys = [GROUP_COL, MONTH_COL]
    return panel.sort([*keys, STEP_COL]).with_columns(
        pl.col("金額").cum_sum().over(keys).alias("cum_金額"),
        pl.col("受注件数").cum_sum().over(keys).alias("cum_受注件数"),
        pl.col("顧客数").cum_sum().over(keys).alias("cum_延べ顧客数"),
        # cum_max であって max ではない。`.max().cum_max()` と書くと
        # 月全体の最大が入り、k日目より後の受注が漏れる（実際に踏んだ）
        pl.col("金額").cum_max().over(keys).alias("cum_最大日次金額"),
    )


def _add_previous_months(panel: pl.DataFrame) -> pl.DataFrame:
    """前月以前に確定した実績を持ち込む。

    ``prev_`` の列はすべて、対象月より前のデータだけから作る。
    月次の着地額は月が終わって初めて確定するので、当月のものは使えない。
    """
    monthly = (
        panel.group_by(GROUP_COL, MONTH_COL)
        .agg(pl.col(TARGET_COL).first().alias("着地"))
        .sort([GROUP_COL, MONTH_COL])
    )
    monthly = monthly.with_columns(
        pl.col("着地").shift(1).over(GROUP_COL).alias("prev_前月の着地"),
        pl.col("着地").shift(1).rolling_mean(3).over(GROUP_COL).alias("prev_直近3ヶ月の平均"),
        pl.col("着地").shift(12).over(GROUP_COL).alias("prev_前年同月の着地"),
        pl.col("着地").shift(13).over(GROUP_COL).alias("prev_前年前月の着地"),
    )
    # 直近3ヶ月の前年同月比。全体の伸びを表す
    monthly = monthly.with_columns(
        (
            pl.col("着地").shift(1).rolling_sum(3).over(GROUP_COL)
            / pl.col("着地").shift(13).rolling_sum(3).over(GROUP_COL)
        ).alias("prev_直近3ヶ月の前年同月比")
    )
    return panel.join(monthly.drop("着地"), on=[GROUP_COL, MONTH_COL], how="left")


def _add_previous_year_progress(panel: pl.DataFrame) -> pl.DataFrame:
    """前年同月の「この時点での進捗率」を持ち込む。

    「例年、10営業日目までに月間の45%が入る」という情報である。
    これがあると、当月の累計を進捗率で割るだけで着地の見込みが出る。
    **この課題で最も効くはずの特徴量**なので、必ず入れる。

    前年同月の営業日数が今年と違うことがある（祝日の位置で1〜2日ずれる）。
    その場合は前年の最終営業日の値、つまり進捗率 1.0 を使う。
    """
    progress = panel.select(
        GROUP_COL,
        pl.col(MONTH_COL).dt.offset_by("12mo").alias(MONTH_COL),
        pl.col(STEP_COL),
        (pl.col("cum_金額") / pl.col(TARGET_COL)).alias("prev_前年同期の進捗率"),
        pl.col("cum_金額").alias("prev_前年同期の累計"),
        pl.col("cum_受注件数").alias("prev_前年同期の受注件数"),
    )
    joined = panel.join(progress, on=[GROUP_COL, MONTH_COL, STEP_COL], how="left")
    # 前年の営業日数が少なく k 日目が存在しない場合は、進捗率 1.0（月が終わっている）
    return joined.with_columns(
        pl.when(
            pl.col("prev_前年同月の着地").is_not_null() & pl.col("prev_前年同期の進捗率").is_null()
        )
        .then(1.0)
        .otherwise(pl.col("prev_前年同期の進捗率"))
        .alias("prev_前年同期の進捗率")
    )


def _add_derived(panel: pl.DataFrame) -> pl.DataFrame:
    """特徴量を作る。**すべて「前年同月の着地額に対する比」にする。**

    最初は金額をそのまま特徴量にしていたが、それでは学習が成立しなかった。
    理由は2つある。

    1. **系列ごとに桁が違う。** 営業2部の月平均が約10億、全社は約38億。
       同じ「累計5億」でも意味がまるで違う。
    2. **水準が年々上がる。** 月平均が 28.6億（2022）→ 46.1億（2024）。
       決定木は学習データの範囲を超える値を**外挿できない**ので、
       検証期間の高い水準を構造的に予測できない。

    金額の列を ``prev_前年同月の着地`` で割って比にすると、両方とも消える。
    系列や年をまたいで同じ尺度になり、木が学んだ規則がそのまま通用する。

    目的変数も同じ理由で比にする（``train`` 側で扱う）。
    """
    scale = pl.col(SCALE_COL)
    remaining = pl.col("cal_月の営業日数") - pl.col(STEP_COL)
    return panel.with_columns(
        # --- カレンダー（もともと尺度を持たない） ---
        remaining.alias("cal_残り営業日数"),
        (pl.col(STEP_COL) / pl.col("cal_月の営業日数")).alias("cal_進捗率"),
        pl.col(MONTH_COL).dt.month().alias("cal_月"),
        pl.col(MONTH_COL).dt.quarter().alias("cal_四半期"),
        # --- 当月の進み具合（前年同月に対する比） ---
        (pl.col("cum_金額") / scale).alias("feat_累計比"),
        (
            pl.col("cum_金額")
            / pl.col("cum_受注件数")
            / (scale / pl.col("prev_前年同期の受注件数"))
        ).alias("feat_平均単価の前年比"),
        (pl.col("cum_受注件数") / pl.col("prev_前年同期の受注件数")).alias("feat_受注件数の前年比"),
        (pl.col("cum_金額") / pl.col("prev_前年同期の累計")).alias("feat_前年同期比"),
        (pl.col("cum_最大日次金額") / scale).alias("feat_最大日次の比率"),
        # --- 手作業の見込み（比で持つ） ---
        (pl.col("cum_金額") / pl.col(STEP_COL) * pl.col("cal_月の営業日数") / scale).alias(
            "feat_線形外挿比"
        ),
        (pl.col("cum_金額") / pl.col("prev_前年同期の進捗率") / scale).alias("feat_進捗率外挿比"),
        (pl.col("cum_金額") / scale + (1.0 - pl.col(STEP_COL) / pl.col("cal_月の営業日数"))).alias(
            "feat_残り按分比"
        ),
        # --- 前年同月の性質 ---
        pl.col("prev_前年同期の進捗率").alias("feat_前年同期の進捗率"),
        # --- 近い過去の水準（前年同月に対する比） ---
        (pl.col("prev_前月の着地") / scale).alias("feat_前月比"),
        (pl.col("prev_直近3ヶ月の平均") / scale).alias("feat_直近3ヶ月平均比"),
        (pl.col("prev_前年前月の着地") / scale).alias("feat_前年前月比"),
        pl.col("prev_直近3ヶ月の前年同月比").alias("feat_直近3ヶ月の伸び率"),
    )


def build_landing_frame(
    transactions: pl.DataFrame, *, include_company: bool = True
) -> pl.DataFrame:
    """取引明細から、着地予測の学習データを作る。

    Args:
        transactions: 取引明細（``受注日``・``部署``・``販売金額`` を含む）。
        include_company: 全社を1つの系列として加えるか。

    Returns:
        1行 = （部署, 対象月, 経過営業日）。``着地額`` が目的変数。
    """
    for column in (_DATE, GROUP_COL, _AMOUNT, "受注番号", "顧客コード"):
        if column not in transactions.columns:
            msg = f"取引明細に必要な列がありません: {column}"
            raise KeyError(msg)

    grid = _business_day_grid(
        transactions.get_column(_DATE).min(), transactions.get_column(_DATE).max()
    )
    daily = _daily_by_group(transactions, include_company=include_company)
    groups = daily.select(GROUP_COL).unique()

    # 受注の無かった営業日も 0 として並べる。抜くと k がずれる
    panel = (
        groups.join(grid, how="cross")
        .join(daily, on=[GROUP_COL, _DATE], how="left")
        .with_columns(
            pl.col("金額").fill_null(0),
            pl.col("受注件数").fill_null(0),
            pl.col("顧客数").fill_null(0),
        )
    )
    # 目的変数: その月の着地額（月全体の合計）
    panel = panel.with_columns(pl.col("金額").sum().over([GROUP_COL, MONTH_COL]).alias(TARGET_COL))

    panel = _add_cumulative(panel)
    panel = _add_previous_months(panel)
    panel = _add_previous_year_progress(panel)
    panel = _add_derived(panel)
    # ベースラインは「円」で比較するので、金額のままの見込みも残す
    panel = panel.with_columns(
        (pl.col("cum_金額") / pl.col(STEP_COL) * pl.col("cal_月の営業日数")).alias("線形外挿"),
        (pl.col("cum_金額") / pl.col("prev_前年同期の進捗率")).alias("進捗率外挿"),
    )

    return panel.sort([GROUP_COL, MONTH_COL, STEP_COL])


def feature_columns(frame: pl.DataFrame) -> list[str]:
    """特徴量として使う列を返す。

    接頭辞で選ぶ。列挙して持つと、特徴量を足したときに追記し忘れて
    **使われないまま気づかない**、あるいは目的変数を混ぜてしまう。

    ``cum_`` と ``prev_`` は中間列であり、特徴量にしない。金額のままなので
    系列や年をまたぐと尺度が変わり、決定木が学べないため（``_add_derived`` 参照）。
    """
    return [
        c for c in frame.columns if c.startswith(FEATURE_PREFIXES) and frame.schema[c].is_numeric()
    ] + [STEP_COL]
