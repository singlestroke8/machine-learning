"""ベースライン予測。

「LightGBM で WAPE 0.31 が出ました」という報告には意味がない。
比較対象がなければ、その数字が良いのか悪いのか誰にも判断できないからである。
需要予測の現場でよく使われる（そして実際そこそこ強い）単純手法を用意し、
必ず一緒に評価する。機械学習モデルがこれらに勝てないなら、
運用コストを払ってまで導入する理由はない。

いずれのベースラインも、学習済みモデルと**まったく同じ特徴量行列**から
計算している。したがって参照している情報も origin 時点までに限られており、
モデルとベースラインの比較が公平になる。
"""

from __future__ import annotations

import numpy as np
import polars as pl

# 表示順を固定するために名前を明示的に持つ
BASELINE_NAMES: tuple[str, ...] = ("naive", "seasonal_naive", "moving_average")

_BASELINE_COLUMNS: dict[str, str] = {
    # origin 当日の実績をそのまま横引きする
    "naive": "org_lag_1",
    # target と同じ曜日の直近実績（週次の季節性を持つ需要に対する定番の手法）
    "seasonal_naive": "org_target_dow_last",
    # 直近の移動平均（窓幅は呼び出し側が決める）
    "moving_average": "org_roll_mean_{window}",
}


def baseline_column(name: str, *, window: int) -> str:
    """ベースラインが参照する特徴量カラム名を返す。

    Raises:
        KeyError: 未知のベースライン名の場合。
    """
    if name not in _BASELINE_COLUMNS:
        msg = f"未知のベースラインです: {name!r}（利用可能: {list(_BASELINE_COLUMNS)}）"
        raise KeyError(msg)
    return _BASELINE_COLUMNS[name].format(window=window)


def compute_baselines(frame: pl.DataFrame, *, window: int) -> dict[str, np.ndarray]:
    """フレームからベースライン予測をまとめて計算する。

    履歴不足で欠損している箇所は 0 で埋める。ベースラインは「何もしない
    運用」の代理なので、値が取れないときに予測を出さないのではなく、
    0 を出したものとして減点しておくほうが実態に近い。

    Args:
        frame: ``build_training_frame`` が返した特徴量行列。
        window: 移動平均ベースラインの窓幅。

    Returns:
        ``{ベースライン名: 予測値の配列}``。参照カラムが無いものは除外される。
    """
    predictions: dict[str, np.ndarray] = {}
    for name in BASELINE_NAMES:
        column = baseline_column(name, window=window)
        if column not in frame.columns:
            continue
        values = frame.get_column(column).fill_null(0.0).to_numpy().astype(np.float64)
        predictions[name] = np.maximum(values, 0.0)
    return predictions
