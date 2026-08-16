"""時系列クロスバリデーションの分割。

``KFold`` のようなランダム分割を時系列に使うと、未来のデータで過去を
予測する形になり、検証スコアが実運用より必ず楽観的になる。ここでは
拡大窓（expanding window）方式を採る。学習期間は常に検証期間より過去にあり、
本番で「過去すべてを使って直近を予測する」状況をそのまま再現する。

なお、特徴量が ``origin = target - horizon`` で打ち切られているため、
学習期間と検証期間の間に追加の gap は原則不要である
（``gap_days`` は、それでも保険をかけたい案件のために残してある）。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class Fold:
    """1つの学習/検証分割。すべて両端を含む日付範囲で表す。"""

    index: int
    train_start: dt.date
    train_end: dt.date
    val_start: dt.date
    val_end: dt.date

    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days + 1

    @property
    def val_days(self) -> int:
        return (self.val_end - self.val_start).days + 1

    def describe(self) -> str:
        return (
            f"fold{self.index}: train {self.train_start}〜{self.train_end}"
            f" ({self.train_days}日) / val {self.val_start}〜{self.val_end}"
            f" ({self.val_days}日)"
        )


def expanding_window_folds(
    dates: Sequence[dt.date] | pl.Series,
    *,
    n_splits: int,
    val_days: int,
    gap_days: int = 0,
) -> list[Fold]:
    """拡大窓方式の分割を、古い順に返す。

    最新の検証期間がデータ末尾に一致するように後ろから区切る。
    直近の期間を必ず検証に含めることで、「今のデータで今どれだけ当たるか」を
    評価できるようにするため。

    Args:
        dates: 対象期間に含まれる日付（重複していてよい）。
        n_splits: 分割数。
        val_days: 1分割あたりの検証日数。
        gap_days: 学習期間と検証期間の間に空ける日数。

    Returns:
        古い順に並んだ ``Fold`` のリスト。

    Raises:
        ValueError: 引数が不正、または期間が短くて分割できない場合。
    """
    if n_splits < 1:
        msg = f"n_splits は1以上である必要があります: {n_splits}"
        raise ValueError(msg)
    if val_days < 1:
        msg = f"val_days は1以上である必要があります: {val_days}"
        raise ValueError(msg)

    values = dates.to_list() if isinstance(dates, pl.Series) else list(dates)
    if not values:
        msg = "日付が空です。"
        raise ValueError(msg)

    start, end = min(values), max(values)

    folds: list[Fold] = []
    for i in range(n_splits):
        # i=0 が最も古い分割になるよう、末尾から数える
        offset = (n_splits - 1 - i) * val_days
        val_end = end - dt.timedelta(days=offset)
        val_start = val_end - dt.timedelta(days=val_days - 1)
        train_end = val_start - dt.timedelta(days=1 + gap_days)

        if train_end < start:
            msg = (
                f"データ期間 {start}〜{end} が短く、"
                f"n_splits={n_splits}, val_days={val_days}, gap_days={gap_days}"
                " では分割できません。"
                " 期間を延ばすか、n_splits / val_days を小さくしてください。"
            )
            raise ValueError(msg)

        folds.append(
            Fold(
                index=i,
                train_start=start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )
        )
    return folds


def split_frame(
    frame: pl.DataFrame, fold: Fold, *, date_col: str = "date"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """フレームを fold の学習部分・検証部分に分ける。"""
    train = frame.filter(
        (pl.col(date_col) >= fold.train_start) & (pl.col(date_col) <= fold.train_end)
    )
    val = frame.filter((pl.col(date_col) >= fold.val_start) & (pl.col(date_col) <= fold.val_end))
    return train, val
