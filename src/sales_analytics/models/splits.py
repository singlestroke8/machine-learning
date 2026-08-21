"""時系列クロスバリデーションの分割。

``KFold`` のようなランダム分割を時系列に使うと、未来のデータで過去を
予測する形になり、検証スコアが実運用より必ず楽観的になる。ここでは
拡大窓（expanding window）方式を採る。学習期間は常に検証期間より過去にあり、
本番で「過去すべてを使って直近を予測する」状況をそのまま再現する。

期間の数え方は**ステップ単位**である。データに現れる日付を昇順に並べ、
その位置で区切る。暦日の加算で区切ると、営業日データでは
「28日ぶん」が実際には20営業日しか含まないことになり、
設定した検証期間と実際の量がずれる。

なお、特徴量が ``origin = target - horizon`` で打ち切られているため、
学習期間と検証期間の間に追加の gap は原則不要である
（``gap_steps`` は、それでも保険をかけたい案件のために残してある）。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class Fold:
    """1つの学習/検証分割。日付範囲は両端を含む。"""

    index: int
    train_start: dt.date
    train_end: dt.date
    val_start: dt.date
    val_end: dt.date
    #: 学習期間に含まれるステップ数（暦日数ではない）
    train_steps: int
    #: 検証期間に含まれるステップ数
    val_steps: int

    def describe(self) -> str:
        return (
            f"fold{self.index}: train {self.train_start}〜{self.train_end}"
            f" ({self.train_steps}ステップ) / val {self.val_start}〜{self.val_end}"
            f" ({self.val_steps}ステップ)"
        )


def expanding_window_folds(
    dates: Sequence[dt.date] | pl.Series,
    *,
    n_splits: int,
    val_steps: int,
    gap_steps: int = 0,
) -> list[Fold]:
    """拡大窓方式の分割を、古い順に返す。

    最新の検証期間がデータ末尾に一致するように後ろから区切る。
    直近の期間を必ず検証に含めることで、「今のデータで今どれだけ当たるか」を
    評価できるようにするため。

    Args:
        dates: 対象期間に含まれる日付（重複していてよい）。
        n_splits: 分割数。
        val_steps: 1分割あたりの検証ステップ数。
        gap_steps: 学習期間と検証期間の間に空けるステップ数。

    Returns:
        古い順に並んだ ``Fold`` のリスト。

    Raises:
        ValueError: 引数が不正、または期間が短くて分割できない場合。
    """
    if n_splits < 1:
        msg = f"n_splits は1以上である必要があります: {n_splits}"
        raise ValueError(msg)
    if val_steps < 1:
        msg = f"val_steps は1以上である必要があります: {val_steps}"
        raise ValueError(msg)

    values = dates.to_list() if isinstance(dates, pl.Series) else list(dates)
    if not values:
        msg = "日付が空です。"
        raise ValueError(msg)

    # データに現れる日付そのものが時間軸になる。
    # 暦日で数えないので、営業日データでも「20ステップ = 20営業日」で揃う。
    timeline = sorted(set(values))
    n_steps = len(timeline)

    folds: list[Fold] = []
    for i in range(n_splits):
        # i=0 が最も古い分割になるよう、末尾から数える
        offset = (n_splits - 1 - i) * val_steps
        val_end_index = n_steps - 1 - offset
        val_start_index = val_end_index - val_steps + 1
        train_end_index = val_start_index - 1 - gap_steps

        if train_end_index < 0:
            msg = (
                f"データ期間 {timeline[0]}〜{timeline[-1]}（{n_steps} ステップ）が短く、"
                f"n_splits={n_splits}, val_steps={val_steps}, gap_steps={gap_steps}"
                " では分割できません。"
                " 期間を延ばすか、n_splits / val_steps を小さくしてください。"
            )
            raise ValueError(msg)

        folds.append(
            Fold(
                index=i,
                train_start=timeline[0],
                train_end=timeline[train_end_index],
                val_start=timeline[val_start_index],
                val_end=timeline[val_end_index],
                train_steps=train_end_index + 1,
                val_steps=val_steps,
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
