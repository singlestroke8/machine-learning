"""誤差の理論下限（ノイズフロア）の推定。

「WAPE 0.286 が出ました」だけでは、その数字が良いのか悪いのか判断できない。
下からはベースラインが、上からはノイズフロアが挟むことで初めて意味を持つ。

    ベースライン (何もしない)  ←──  モデル  ──→  理論下限 (これ以上は不可能)

需要は確率的に発生するので、真の平均需要 λ を完全に知っていても、
実際に何個売れるかは当てられない。この「原理的に取り除けない誤差」が下限になる。
モデルの WAPE が下限に近ければ、学習できる構造はもう出し尽くしている。
そのときにハイパーパラメータを追い込んでも成果は出ないので、
「特徴量を増やす」ではなく「そもそも別の情報源を取りに行く」判断に切り替えられる。

この推定ができるのは、データが合成であり真の λ が分かるからである。
実データではこの手が使えないため、代わりに
「同一条件の繰り返し観測のばらつき」などから見積もることになる。
"""

from __future__ import annotations

import numpy as np

from demand_forecast.config import DataConfig
from demand_forecast.data.generate import OVERDISPERSION_SHAPE, generate_demand_data
from demand_forecast.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_N_SAMPLES = 3000
DEFAULT_N_DRAWS = 3000


def estimate_noise_floor(
    cfg: DataConfig,
    *,
    seed: int = 42,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_draws: int = DEFAULT_N_DRAWS,
) -> dict[str, float]:
    """真の期待需要を知る「予言者」でも避けられない WAPE を推定する。

    各日の需要は負の二項分布（ガンマ・ポアソン混合）に従う。
    WAPE は絶対誤差の指標なので、これを最小化する予測は**条件付き中央値**である。
    そこで λ ごとに分布からサンプリングし、中央値を予測としたときの
    平均絶対誤差を集計して Σ|誤差| / Σ実績 を求める。

    Args:
        cfg: データ生成設定。
        seed: 生成シード（学習に使ったものと合わせること）。
        n_samples: 評価に使う (系列, 日) の標本数。
        n_draws: 1標本あたりのモンテカルロ試行数。

    Returns:
        ``oracle_wape``（理論下限）と、推定に使った標本数などを含む辞書。
    """
    demand = generate_demand_data(cfg, seed=seed, include_expected=True)
    lam_all = demand.get_column("expected_demand").to_numpy().astype(np.float64)

    rng = np.random.default_rng(seed)
    size = min(n_samples, lam_all.size)
    sampled = lam_all[rng.choice(lam_all.size, size=size, replace=False)]

    total_abs_error = 0.0
    total_actual = 0.0
    for lam in sampled:
        # ガンマ・ポアソン混合からのサンプリング = 負の二項分布からのサンプリング
        overdispersion = rng.gamma(OVERDISPERSION_SHAPE, 1.0 / OVERDISPERSION_SHAPE, size=n_draws)
        draws = rng.poisson(lam * overdispersion)
        total_abs_error += float(np.abs(draws - np.median(draws)).mean())
        total_actual += float(draws.mean())

    oracle_wape = total_abs_error / total_actual
    logger.info("理論下限 WAPE ≈ %.4f（標本 %d 点）", oracle_wape, size)

    return {
        "oracle_wape": oracle_wape,
        "n_samples": float(size),
        "n_draws": float(n_draws),
        "mean_expected_demand": float(lam_all.mean()),
    }


def explain_gap(model_wape: float, baseline_wape: float, oracle_wape: float) -> dict[str, float]:
    """モデルが「学習可能な余地」のどれだけを取れているかを計算する。

    ベースラインから理論下限までの区間が、モデルに取れる余地のすべてである。
    その何割を実際に取れたかを見ることで、
    「あと何%改善できるのか」を具体的な数字で語れるようになる。

    Args:
        model_wape: モデルの WAPE。
        baseline_wape: 比較対象のベースラインの WAPE。
        oracle_wape: 理論下限の WAPE。

    Returns:
        改善率と「取り切った割合」を含む辞書。

    Raises:
        ValueError: ベースラインが理論下限を下回っている場合（前提が壊れている）。
    """
    learnable = baseline_wape - oracle_wape
    if learnable <= 0:
        msg = (
            f"ベースライン({baseline_wape:.4f}) が理論下限({oracle_wape:.4f}) を"
            " 下回っています。下限の推定かベースラインの計算を見直してください。"
        )
        raise ValueError(msg)

    captured = baseline_wape - model_wape
    return {
        "improvement_over_baseline": captured / baseline_wape,
        "captured_share_of_learnable": captured / learnable,
        "remaining_gap": model_wape - oracle_wape,
    }
