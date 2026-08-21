"""評価指標のテスト。

指標の実装ミスは「モデルが良くなった/悪くなった」の判断そのものを狂わせる。
手で答えが出せる例で固めておく。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sales_analytics.models.metrics import (
    bias,
    coverage,
    evaluate_point_forecast,
    evaluate_quantile_forecast,
    mae,
    pinball_loss,
    rmse,
    smape,
    wape,
)


def test_perfect_forecast_scores_zero() -> None:
    y = [10.0, 20.0, 30.0]
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert wape(y, y) == 0.0
    assert bias(y, y) == 0.0


def test_wape_is_total_error_over_total_actual() -> None:
    """WAPE = Σ|誤差| / Σ実績 であること。"""
    y_true = [10.0, 0.0, 90.0]
    y_pred = [12.0, 3.0, 85.0]
    # |2| + |3| + |5| = 10, Σy = 100
    assert wape(y_true, y_pred) == pytest.approx(0.10)


def test_wape_survives_zero_actuals() -> None:
    """実績0の日が含まれても発散しないこと（MAPE との決定的な差）。"""
    assert math.isfinite(wape([0.0, 10.0], [3.0, 10.0]))


def test_wape_is_nan_when_all_actuals_are_zero() -> None:
    assert math.isnan(wape([0.0, 0.0], [1.0, 2.0]))


def test_bias_sign_indicates_direction() -> None:
    """過大予測なら正、過小予測なら負になること。"""
    assert bias([10.0, 10.0], [12.0, 12.0]) == pytest.approx(0.2)
    assert bias([10.0, 10.0], [8.0, 8.0]) == pytest.approx(-0.2)


def test_rmse_penalizes_large_errors_more_than_mae() -> None:
    y_true = [0.0, 0.0, 0.0, 0.0]
    spread = [0.0, 0.0, 0.0, 8.0]
    even = [2.0, 2.0, 2.0, 2.0]
    assert mae(y_true, spread) == pytest.approx(mae(y_true, even))
    assert rmse(y_true, spread) > rmse(y_true, even)


def test_smape_is_bounded() -> None:
    assert 0.0 <= smape([1.0, 5.0], [3.0, 2.0]) <= 2.0


def test_pinball_loss_penalizes_asymmetrically() -> None:
    """高い分位点では過小予測のほうが重く罰せられること。"""
    y_true = [10.0]
    under = pinball_loss(y_true, [8.0], quantile=0.9)
    over = pinball_loss(y_true, [12.0], quantile=0.9)
    assert under > over

    # q=0.5 では対称になり、MAE の半分に一致する
    assert pinball_loss(y_true, [8.0], 0.5) == pytest.approx(mae(y_true, [8.0]) / 2)


def test_pinball_loss_rejects_invalid_quantile() -> None:
    with pytest.raises(ValueError, match="分位点"):
        pinball_loss([1.0], [1.0], quantile=1.0)


def test_coverage_counts_inclusion() -> None:
    y_true = [5.0, 15.0, 25.0]
    assert coverage(y_true, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]) == pytest.approx(1 / 3)
    assert coverage(y_true, [0.0] * 3, [30.0] * 3) == 1.0


def test_evaluate_point_forecast_returns_expected_keys() -> None:
    metrics = evaluate_point_forecast([1.0, 2.0], [1.5, 1.5])
    assert set(metrics) == {"wape", "mae", "rmse", "smape", "bias"}


def test_evaluate_quantile_forecast_includes_interval_coverage() -> None:
    y_true = np.array([10.0, 20.0])
    predictions = {
        0.1: np.array([5.0, 15.0]),
        0.5: np.array([10.0, 19.0]),
        0.9: np.array([15.0, 25.0]),
    }
    metrics = evaluate_quantile_forecast(y_true, dict(predictions))
    assert metrics["interval_coverage"] == 1.0
    assert metrics["interval_nominal"] == pytest.approx(0.8)
    assert "pinball_q0.1" in metrics
    assert metrics["wape"] == pytest.approx(1 / 30)


def test_evaluate_quantile_forecast_requires_median() -> None:
    with pytest.raises(KeyError):
        evaluate_quantile_forecast([1.0], {0.9: [1.0]})


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="長さが一致しません"):
        mae([1.0, 2.0], [1.0])


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="空です"):
        mae([], [])
