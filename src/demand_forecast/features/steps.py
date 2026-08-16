"""時間軸を「ステップ」として扱うための対応づけ。

このプロジェクトでは、時間の刻みを**暦日ではなくステップ**で数える。

- 小売のように毎日売上が立つデータ: 1 ステップ = 1 暦日
- 法人取引のように土日祝に受注が無いデータ: 1 ステップ = 1 営業日

ラグや移動集計は元々「行の位置」で動くので、どちらでも正しく働く。
問題になるのは origin 行と target 行を horizon ぶんずらして結合するところで、
ここを暦日の加算（``+14d``）で書くと、営業日軸では土日を跨いだ瞬間にずれる。

そこで日付に連番（ステップ）を振り、**結合をステップの加算で行う**。
こうすると、暦日でも営業日でも同じコードが正しく動く。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import polars as pl

from demand_forecast.config import CalendarMode
from demand_forecast.features.calendar import business_days

STEP_COL = "step"


def build_step_index(dates: Sequence[dt.date]) -> dict[dt.date, int]:
    """日付を昇順の連番に対応づける。

    データに現れる日付だけを対象にするので、
    営業日データなら「営業日の連番」が自然に得られる。

    Raises:
        ValueError: 日付が1つも無い場合。
    """
    unique_dates = sorted(set(dates))
    if not unique_dates:
        msg = "日付が空です。ステップを振れません。"
        raise ValueError(msg)
    return {date: index for index, date in enumerate(unique_dates)}


def build_timeline(start: dt.date, end: dt.date, calendar: CalendarMode) -> list[dt.date]:
    """期間から、欠けのない完全な時間軸を作る。

    推論時はこちらを使う。履歴と予測対象に現れる日付だけで連番を振ると、
    間の日が抜けている場合に horizon を取り違える。
    たとえば「3営業日先と7営業日先だけ予測してほしい」という要求では、
    2つの対象日が隣り合ったステップとみなされ、どちらも近い将来として扱われてしまう。
    """
    if calendar == "business":
        return business_days(start, end)

    days: list[dt.date] = []
    current = start
    while current <= end:
        days.append(current)
        current += dt.timedelta(days=1)
    return days


def add_step_column(
    df: pl.DataFrame,
    step_index: dict[dt.date, int],
    *,
    date_col: str = "date",
) -> pl.DataFrame:
    """日付からステップ列を付与する。

    対応表に無い日付があれば、黙って欠損にせず例外にする。
    推論時に「学習時の時間軸に無い日」を渡された場合、
    静かに落ちるより明示的に落ちたほうが原因を追いやすい。

    Raises:
        ValueError: 対応表に無い日付が含まれる場合。
    """
    unknown = set(df.get_column(date_col).unique().to_list()) - set(step_index)
    if unknown:
        msg = f"ステップ対応表に無い日付です: {sorted(str(d) for d in unknown)[:5]}"
        raise ValueError(msg)

    return df.with_columns(
        pl.col(date_col).replace_strict(step_index, return_dtype=pl.Int32).alias(STEP_COL)
    )
