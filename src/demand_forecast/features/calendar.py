"""カレンダー特徴量。

カレンダー由来の特徴量は「予測対象日の情報」でありながら、
予測時点で確定している数少ない情報である。したがってラグ特徴量と違い、
horizon による打ち切りを受けない。

なお、経過日数のような「生の時間インデックス」は特徴量に含めていない。
勾配ブースティング木は学習データの外側に外挿できないため、時間インデックスを
入れると将来区間で必ず端の値に張り付き、トレンドを取り違える。
水準とトレンドはラグ・移動平均特徴量に担わせる方針をとっている。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import numpy as np
import polars as pl

# 日付固定の祝日（月, 日）
_FIXED_HOLIDAYS: tuple[tuple[int, int], ...] = (
    (1, 1),  # 元日
    (2, 11),  # 建国記念の日
    (2, 23),  # 天皇誕生日（2020年以降）
    (4, 29),  # 昭和の日
    (5, 3),  # 憲法記念日
    (5, 4),  # みどりの日
    (5, 5),  # こどもの日
    (8, 11),  # 山の日
    (11, 3),  # 文化の日
    (11, 23),  # 勤労感謝の日
)

# ハッピーマンデー制度の祝日（月, 第何週の月曜か）
_NTH_MONDAY_HOLIDAYS: tuple[tuple[int, int], ...] = (
    (1, 2),  # 成人の日
    (7, 3),  # 海の日
    (9, 3),  # 敬老の日
    (10, 2),  # スポーツの日
)


def _nth_monday(year: int, month: int, nth: int) -> dt.date:
    """指定した月の第 ``nth`` 月曜日を返す。"""
    first = dt.date(year, month, 1)
    offset = (0 - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (nth - 1))


def _equinox_days(year: int) -> tuple[int, int]:
    """春分の日・秋分の日を近似計算する（1980〜2099年で有効）。"""
    spring = int(20.8431 + 0.242194 * (year - 1980) - (year - 1980) // 4)
    autumn = int(23.2488 + 0.242194 * (year - 1980) - (year - 1980) // 4)
    return spring, autumn


def _national_holidays(year: int) -> set[dt.date]:
    """指定年の祝日集合（振替休日を含む）を返す。"""
    holidays = {dt.date(year, m, d) for m, d in _FIXED_HOLIDAYS}
    holidays |= {_nth_monday(year, m, n) for m, n in _NTH_MONDAY_HOLIDAYS}

    spring, autumn = _equinox_days(year)
    holidays.add(dt.date(year, 3, spring))
    holidays.add(dt.date(year, 9, autumn))

    # 振替休日: 日曜と重なった祝日は翌月曜が休みになる
    substitutes = set()
    for holiday in holidays:
        if holiday.weekday() == 6:  # 日曜
            substitute = holiday + dt.timedelta(days=1)
            while substitute in holidays:
                substitute += dt.timedelta(days=1)
            substitutes.add(substitute)
    return holidays | substitutes


def japanese_holiday_flags(dates: Sequence[dt.date]) -> np.ndarray:
    """日付列に対する休日フラグ（int8 配列）を返す。

    法定の祝日に加えて、年末年始（12/30〜1/3）も休日として扱う。
    小売の需要という観点では、法定祝日かどうかよりも
    「店舗と客の生活リズムが平日と違うか」のほうが説明力が高いため。
    """
    if not dates:
        return np.zeros(0, dtype=np.int8)

    years = {d.year for d in dates}
    holiday_set: set[dt.date] = set()
    for year in years:
        holiday_set |= _national_holidays(year)

    flags = np.zeros(len(dates), dtype=np.int8)
    for i, d in enumerate(dates):
        is_year_end = (d.month == 12 and d.day >= 30) or (d.month == 1 and d.day <= 3)
        if d in holiday_set or is_year_end:
            flags[i] = 1
    return flags


def add_calendar_features(
    df: pl.DataFrame,
    *,
    date_col: str = "date",
    fourier_order: int = 3,
) -> pl.DataFrame:
    """予測対象日のカレンダー特徴量を付与する。

    Args:
        df: ``date_col`` を含む DataFrame。
        date_col: 日付カラム名。
        fourier_order: 年周期フーリエ項の次数。0 なら付与しない。

    Returns:
        カレンダー特徴量を追加した新しい DataFrame（入力は変更しない）。
    """
    dates: list[dt.date] = df.get_column(date_col).to_list()
    holiday = japanese_holiday_flags(dates)

    out = df.with_columns(
        pl.col(date_col).dt.weekday().cast(pl.Int8).alias("cal_dow"),
        pl.col(date_col).dt.month().cast(pl.Int8).alias("cal_month"),
        pl.col(date_col).dt.day().cast(pl.Int8).alias("cal_day"),
        pl.col(date_col).dt.week().cast(pl.Int8).alias("cal_week"),
        pl.col(date_col).dt.ordinal_day().cast(pl.Int16).alias("cal_doy"),
        pl.Series("cal_is_holiday", holiday, dtype=pl.Int8),
    ).with_columns(
        (pl.col("cal_dow") >= 6).cast(pl.Int8).alias("cal_is_weekend"),
        # 給料日前後は小売需要が動きやすいので、月内の位置も明示的に持たせる
        (pl.col("cal_day") <= 5).cast(pl.Int8).alias("cal_is_month_start"),
        (pl.col("cal_day") >= 25).cast(pl.Int8).alias("cal_is_month_end"),
    )

    for k in range(1, fourier_order + 1):
        angle = 2.0 * np.pi * k * pl.col("cal_doy").cast(pl.Float64) / 365.25
        out = out.with_columns(
            angle.sin().alias(f"cal_yearly_sin_{k}"),
            angle.cos().alias(f"cal_yearly_cos_{k}"),
        )

    return out
