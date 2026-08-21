"""時系列CV分割のテスト。

「学習期間が検証期間より必ず過去にある」ことが崩れると、
評価が静かに楽観的になる。そこを重点的に確認する。
"""

from __future__ import annotations

import datetime as dt
from itertools import pairwise

import polars as pl
import pytest

from sales_analytics.models.splits import expanding_window_folds, split_frame


@pytest.fixture
def dates() -> list[dt.date]:
    start = dt.date(2025, 1, 1)
    return [start + dt.timedelta(days=i) for i in range(365)]


def test_folds_are_returned_oldest_first(dates: list[dt.date]) -> None:
    folds = expanding_window_folds(dates, n_splits=3, val_steps=28)
    assert [f.index for f in folds] == [0, 1, 2]
    assert folds[0].val_start < folds[1].val_start < folds[2].val_start


def test_train_never_overlaps_validation(dates: list[dt.date]) -> None:
    """学習期間の終わりが検証期間の開始より前であること。"""
    for fold in expanding_window_folds(dates, n_splits=4, val_steps=21):
        assert fold.train_end < fold.val_start


def test_training_window_expands(dates: list[dt.date]) -> None:
    """後の fold ほど学習期間が長くなること（拡大窓）。"""
    folds = expanding_window_folds(dates, n_splits=3, val_steps=28)
    lengths = [f.train_steps for f in folds]
    assert lengths == sorted(lengths)
    assert len({f.train_start for f in folds}) == 1  # 開始日は共通


def test_last_fold_ends_at_data_end(dates: list[dt.date]) -> None:
    """最新の検証期間がデータ末尾に一致すること。"""
    folds = expanding_window_folds(dates, n_splits=3, val_steps=28)
    assert folds[-1].val_end == max(dates)


def test_validation_windows_do_not_overlap(dates: list[dt.date]) -> None:
    folds = expanding_window_folds(dates, n_splits=3, val_steps=28)
    for earlier, later in pairwise(folds):
        assert earlier.val_end < later.val_start


def test_gap_steps_are_respected(dates: list[dt.date]) -> None:
    gap = 7
    for fold in expanding_window_folds(dates, n_splits=2, val_steps=28, gap_steps=gap):
        assert (fold.val_start - fold.train_end).days == gap + 1


def test_val_steps_matches_configuration(dates: list[dt.date]) -> None:
    for fold in expanding_window_folds(dates, n_splits=3, val_steps=14):
        assert fold.val_steps == 14


def test_too_short_period_raises(dates: list[dt.date]) -> None:
    with pytest.raises(ValueError, match="短く"):
        expanding_window_folds(dates, n_splits=20, val_steps=28)


def test_invalid_arguments_are_rejected(dates: list[dt.date]) -> None:
    with pytest.raises(ValueError, match="n_splits"):
        expanding_window_folds(dates, n_splits=0, val_steps=28)
    with pytest.raises(ValueError, match="val_steps"):
        expanding_window_folds(dates, n_splits=1, val_steps=0)
    with pytest.raises(ValueError, match="空です"):
        expanding_window_folds([], n_splits=1, val_steps=28)


def test_accepts_polars_series(dates: list[dt.date]) -> None:
    folds = expanding_window_folds(pl.Series("date", dates), n_splits=2, val_steps=28)
    assert len(folds) == 2


def test_split_frame_selects_correct_rows(dates: list[dt.date]) -> None:
    frame = pl.DataFrame({"date": dates, "value": range(len(dates))}).with_columns(
        pl.col("date").cast(pl.Date)
    )
    fold = expanding_window_folds(dates, n_splits=2, val_steps=28)[0]
    train, val = split_frame(frame, fold)

    assert train.get_column("date").max() == fold.train_end
    assert val.get_column("date").min() == fold.val_start
    assert val.height == fold.val_steps
    assert train.height + val.height <= frame.height
