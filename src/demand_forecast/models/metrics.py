"""評価指標。

主要指標は **WAPE**（重み付き絶対パーセント誤差）を採用している。
理由は ``docs/adr/0006-primary-metric-wape.md`` に書いたが、要点は
「MAPE は実需が小さい日に爆発するので、需要が薄いSKUを含む在庫の意思決定に
使えない」ことと、「WAPE は分母が総需要なので、そのまま
『在庫全体で何%外したか』というビジネス上の言葉に翻訳できる」ことにある。

バイアス（systematic over/under forecast）も必ず併記する。在庫の文脈では
「平均的に何%外すか」より「上振れに偏っているか下振れに偏っているか」の
ほうが、欠品と過剰在庫のどちらに効くかを直接決めるため。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_float_arrays(y_true: object, y_pred: object) -> tuple[FloatArray, FloatArray]:
    """入力を float 配列に揃え、長さの一致を確認する。"""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        msg = f"y_true と y_pred の長さが一致しません: {yt.shape} vs {yp.shape}"
        raise ValueError(msg)
    if yt.size == 0:
        msg = "評価対象が空です。"
        raise ValueError(msg)
    return yt, yp


def mae(y_true: object, y_pred: object) -> float:
    """平均絶対誤差。"""
    yt, yp = _as_float_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: object, y_pred: object) -> float:
    """二乗平均平方根誤差。"""
    yt, yp = _as_float_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def wape(y_true: object, y_pred: object) -> float:
    """重み付き絶対パーセント誤差 = Σ|y - ŷ| / Σ|y|。

    実需の総量で正規化するため、需要ゼロの日が含まれていても発散しない。
    総需要が 0 の場合のみ ``nan`` を返す。
    """
    yt, yp = _as_float_arrays(y_true, y_pred)
    denominator = float(np.sum(np.abs(yt)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / denominator)


def bias(y_true: object, y_pred: object) -> float:
    """予測の偏り = Σ(ŷ - y) / Σy。正なら過大予測、負なら過小予測。"""
    yt, yp = _as_float_arrays(y_true, y_pred)
    denominator = float(np.sum(yt))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(yp - yt) / denominator)


def smape(y_true: object, y_pred: object) -> float:
    """対称平均絶対パーセント誤差（0〜2 のスケール）。"""
    yt, yp = _as_float_arrays(y_true, y_pred)
    denominator = np.abs(yt) + np.abs(yp)
    mask = denominator > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(yt[mask] - yp[mask]) / denominator[mask]))


def pinball_loss(y_true: object, y_pred: object, quantile: float) -> float:
    """ピンボール損失（分位点回帰の評価指標）。

    Args:
        quantile: 評価する分位点（0 < q < 1）。

    Raises:
        ValueError: 分位点が範囲外の場合。
    """
    if not 0.0 < quantile < 1.0:
        msg = f"分位点は 0 < q < 1 である必要があります: {quantile}"
        raise ValueError(msg)
    yt, yp = _as_float_arrays(y_true, y_pred)
    delta = yt - yp
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1.0) * delta)))


def coverage(y_true: object, lower: object, upper: object) -> float:
    """予測区間が実績を含んでいた割合。

    公称のカバー率（例: q=0.1〜0.9 なら 0.8）と比べることで、
    区間が楽観的すぎないかを点検する。
    """
    yt, lo = _as_float_arrays(y_true, lower)
    _, hi = _as_float_arrays(y_true, upper)
    return float(np.mean((yt >= lo) & (yt <= hi)))


def evaluate_point_forecast(y_true: object, y_pred: object) -> dict[str, float]:
    """点予測の指標一式を返す。"""
    return {
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }


def evaluate_quantile_forecast(
    y_true: object,
    predictions: dict[float, object],
) -> dict[str, float]:
    """分位点予測の指標一式を返す。

    Args:
        y_true: 実績。
        predictions: ``{分位点: 予測値}`` の辞書。0.5 を必ず含むこと。

    Raises:
        KeyError: 0.5 が含まれていない場合。
    """
    if 0.5 not in predictions:
        msg = "点予測に用いる分位点 0.5 が含まれていません。"
        raise KeyError(msg)

    metrics = evaluate_point_forecast(y_true, predictions[0.5])
    for q, pred in sorted(predictions.items()):
        metrics[f"pinball_q{q:g}"] = pinball_loss(y_true, pred, q)

    quantiles = sorted(predictions)
    lo_q, hi_q = quantiles[0], quantiles[-1]
    if lo_q < hi_q:
        metrics["interval_coverage"] = coverage(y_true, predictions[lo_q], predictions[hi_q])
        metrics["interval_nominal"] = hi_q - lo_q
    return metrics
