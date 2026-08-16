"""理論下限の推定のテスト。"""

from __future__ import annotations

import datetime as dt

import pytest

from demand_forecast.analysis import estimate_noise_floor, explain_gap
from demand_forecast.config import DataConfig
from demand_forecast.data.generate import generate_demand_data

TINY_CONFIG = DataConfig(
    start_date=dt.date(2025, 1, 1),
    end_date=dt.date(2025, 2, 28),
    n_stores=1,
    n_skus=2,
)


def test_expected_demand_is_hidden_by_default() -> None:
    """真の期待需要は既定では返さないこと（学習に混入させないため）。"""
    df = generate_demand_data(TINY_CONFIG, seed=1)
    assert "expected_demand" not in df.columns


def test_expected_demand_is_returned_when_requested() -> None:
    df = generate_demand_data(TINY_CONFIG, seed=1, include_expected=True)
    assert "expected_demand" in df.columns
    assert df.get_column("expected_demand").min() > 0


def test_including_expected_does_not_change_other_columns() -> None:
    """列を1つ増やしても、生成される需要そのものは変わらないこと。"""
    without = generate_demand_data(TINY_CONFIG, seed=1)
    with_expected = generate_demand_data(TINY_CONFIG, seed=1, include_expected=True)
    assert without.equals(with_expected.drop("expected_demand"))


def test_noise_floor_is_a_plausible_wape() -> None:
    """推定される下限が、WAPE として妥当な範囲に収まること。"""
    result = estimate_noise_floor(TINY_CONFIG, seed=1, n_samples=200, n_draws=400)
    assert 0.0 < result["oracle_wape"] < 1.0
    assert result["mean_expected_demand"] > 0


def test_noise_floor_sample_size_is_capped_by_data() -> None:
    """データ行数より多い標本数を指定しても、行数で頭打ちになること。"""
    n_rows = generate_demand_data(TINY_CONFIG, seed=1).height
    result = estimate_noise_floor(TINY_CONFIG, seed=1, n_samples=n_rows * 10, n_draws=100)
    assert result["n_samples"] == n_rows


def test_noise_floor_is_stable_across_runs() -> None:
    """同じシードなら推定値がぶれないこと。"""
    first = estimate_noise_floor(TINY_CONFIG, seed=3, n_samples=150, n_draws=300)
    second = estimate_noise_floor(TINY_CONFIG, seed=3, n_samples=150, n_draws=300)
    assert first["oracle_wape"] == pytest.approx(second["oracle_wape"])


def test_explain_gap_computes_captured_share() -> None:
    """ベースラインと下限のあいだで、モデルがどれだけ取れたかを計算できること。"""
    gap = explain_gap(model_wape=0.30, baseline_wape=0.40, oracle_wape=0.28)
    assert gap["improvement_over_baseline"] == pytest.approx(0.25)
    # 学習可能な余地 0.12 のうち 0.10 を回収
    assert gap["captured_share_of_learnable"] == pytest.approx(0.10 / 0.12)
    assert gap["remaining_gap"] == pytest.approx(0.02)


def test_explain_gap_rejects_impossible_baseline() -> None:
    """ベースラインが理論下限を下回る（ありえない）入力を弾くこと。"""
    with pytest.raises(ValueError, match="下回っています"):
        explain_gap(model_wape=0.30, baseline_wape=0.20, oracle_wape=0.28)
