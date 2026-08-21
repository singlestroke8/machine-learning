"""着地予測のベースライン。

**モデルの成績は、単独では意味を持たない。** 比較相手が弱ければ、
どんな手法でも良く見える。ここでは手作業でできる方法のうち、
**最も強いもの**まで用意する。

    「これに勝てないなら、機械学習は要らない」

という判断ができる状態にしておくのが目的である。
特に ``進捗率外挿`` は、この課題で機械学習にやらせようとしていることの
手作業版にあたる。ここに勝てるかどうかが本質的な問いになる。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from sales_analytics.tasks.landing.dataset import STEP_COL


@dataclass(frozen=True)
class Baseline:
    """ベースライン1つ分。"""

    name: str
    description: str
    expression: Callable[[], pl.Expr]

    @property
    def column(self) -> str:
        return f"base_{self.name}"


def _year_over_year() -> pl.Expr:
    """前年同月 × 直近の伸び率。

    **経過日数を一切使わない。** 月初に出しても月末に出しても同じ数字を返す。
    これが、この方法の限界であり、機械学習を使う理由でもある。
    """
    growth = pl.col("prev_直近3ヶ月の前年同月比").fill_null(1.0)
    return pl.col("prev_前年同月の着地") * growth


def _linear() -> pl.Expr:
    """今のペースが月末まで続くと仮定する。

    月の前半ほど外す。数営業日の実績を20日ぶんに引き伸ばすため。
    """
    return pl.col("線形外挿")


def _remaining_split_expr() -> pl.Expr:
    """当月の累計 ＋ 前年同月の「残り日数ぶん」を按分して足す。

    残りを日数比で按分する。月内の受注が均等に発生するという仮定が入っている。
    実際には月末に寄るので、その分だけ甘くなる。
    """
    remaining_ratio = 1.0 - pl.col(STEP_COL) / pl.col("cal_月の営業日数")
    return pl.col("cum_金額") + pl.col("prev_前年同月の着地") * remaining_ratio


def _remaining_actual() -> pl.Expr:
    """当月の累計 ＋ 前年同月の「残り日数に実際に入った額」。

    ``残り按分`` は月内の受注が均等に発生すると仮定するが、実際は月末に寄る。
    前年同月の実額を使えばその形を織り込める。**手作業でできる方法として
    最も筋が良い**ので、これを本命の比較相手にする。

    弱いベースラインを選べば改善率はいくらでも大きく書けるが、
    それは数字の作り方の問題であって実力ではない。
    """
    return pl.col("cum_金額") + pl.col("prev_前年同月の着地") - pl.col("prev_前年同期の累計")


def _progress_ratio() -> pl.Expr:
    """当月の累計 ÷ 前年同月の「この時点での進捗率」。

    「例年この時点で45%入っているのに、今年は40%しか入っていない」
    という情報を使う。**月内の受注が月末に寄ることも織り込める**ので、
    手作業でできる方法としては最も筋が良い。

    実質的な本命の比較相手であり、機械学習がやろうとしていることの手作業版。
    """
    return pl.col("進捗率外挿")


BASELINES: tuple[Baseline, ...] = (
    Baseline("前年同月", "前年同月 × 直近3ヶ月の伸び率。経過日数を使わない", _year_over_year),
    Baseline("線形外挿", "今のペースが月末まで続くと仮定する", _linear),
    Baseline("残り按分", "当月の累計 ＋ 前年同月を残り日数で按分", _remaining_split_expr),
    Baseline("進捗率外挿", "当月の累計 ÷ 前年同期の進捗率", _progress_ratio),
    Baseline("残り実額", "当月の累計 ＋ 前年同月の残り実額。最も強い手作業版", _remaining_actual),
)


def add_baselines(frame: pl.DataFrame) -> pl.DataFrame:
    """すべてのベースラインの予測値を列として付ける。

    前年同月が無い期間（データの最初の12ヶ月）は欠損のままにする。
    ゼロで埋めると、その行だけベースラインが極端に外して見え、
    **モデルが不当に強く見える**。
    """
    return frame.with_columns([b.expression().alias(b.column) for b in BASELINES])


def baseline_columns() -> list[str]:
    return [b.column for b in BASELINES]
