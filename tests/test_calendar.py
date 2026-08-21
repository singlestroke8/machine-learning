"""カレンダー特徴量のテスト。"""

from __future__ import annotations

import datetime as dt

import polars as pl

from sales_analytics.features.calendar import add_calendar_features, japanese_holiday_flags


def test_fixed_holidays_are_detected() -> None:
    """日付固定の祝日が検出されること。"""
    dates = [dt.date(2026, 1, 1), dt.date(2026, 5, 3), dt.date(2026, 11, 3)]
    assert japanese_holiday_flags(dates).tolist() == [1, 1, 1]


def test_happy_monday_is_detected() -> None:
    """ハッピーマンデーの祝日が正しい日に立つこと。

    2026年の成人の日は1月第2月曜（1月12日）。
    """
    flags = japanese_holiday_flags([dt.date(2026, 1, 12), dt.date(2026, 1, 13)])
    assert flags.tolist() == [1, 0]


def test_substitute_holiday_is_detected() -> None:
    """日曜と重なった祝日の振替休日が翌月曜に立つこと。

    2026年5月3日（憲法記念日）は日曜。5月4・5日も祝日なので、
    振替は5月6日（水）になる。
    """
    dates = [dt.date(2026, 5, d) for d in range(3, 8)]
    assert japanese_holiday_flags(dates).tolist() == [1, 1, 1, 1, 0]


def test_year_end_period_is_treated_as_holiday() -> None:
    """年末年始が休日扱いになること（小売の需要という観点での判断）。"""
    dates = [dt.date(2025, 12, 29), dt.date(2025, 12, 30), dt.date(2026, 1, 3), dt.date(2026, 1, 4)]
    assert japanese_holiday_flags(dates).tolist() == [0, 1, 1, 0]


def test_empty_input_returns_empty_array() -> None:
    assert japanese_holiday_flags([]).size == 0


def test_calendar_features_are_added() -> None:
    """想定したカラムが、想定した値で付与されること。"""
    df = pl.DataFrame({"date": [dt.date(2026, 8, 15)]}).with_columns(pl.col("date").cast(pl.Date))
    out = add_calendar_features(df, fourier_order=2)

    assert out.get_column("cal_dow").item() == 6  # 土曜（Polars は月曜=1）
    assert out.get_column("cal_is_weekend").item() == 1
    assert out.get_column("cal_month").item() == 8
    assert out.get_column("cal_doy").item() == 227
    assert out.get_column("cal_is_month_start").item() == 0
    for k in (1, 2):
        assert f"cal_yearly_sin_{k}" in out.columns
        assert f"cal_yearly_cos_{k}" in out.columns


def test_fourier_order_zero_adds_no_terms() -> None:
    df = pl.DataFrame({"date": [dt.date(2026, 8, 15)]}).with_columns(pl.col("date").cast(pl.Date))
    out = add_calendar_features(df, fourier_order=0)
    assert not [c for c in out.columns if c.startswith("cal_yearly_")]


def test_input_frame_is_not_mutated() -> None:
    df = pl.DataFrame({"date": [dt.date(2026, 8, 15)]}).with_columns(pl.col("date").cast(pl.Date))
    before = df.columns.copy()
    add_calendar_features(df)
    assert df.columns == before
