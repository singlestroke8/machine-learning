"""Optuna によるハイパーパラメータ探索。

探索は**点予測（q=0.5）の WAPE のみ**を対象にする。3本の分位点すべてを
毎試行で学習すると計算量が3倍になる一方、木の構造に関するパラメータは
分位点間でほぼ共通に効くため、割に合わないからである。

探索も評価と同じ拡大窓CVで行う。ここでランダム分割を使うと、
「探索の段階でだけ未来を覗いた」モデルが選ばれてしまう。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from demand_forecast.config import Config
from demand_forecast.data.loaders import read_demand_frame
from demand_forecast.features.pipeline import (
    SeriesEncoder,
    build_training_frame,
    feature_columns,
)
from demand_forecast.logging_utils import get_logger
from demand_forecast.models.estimator import QuantileForecaster
from demand_forecast.models.metrics import wape
from demand_forecast.models.splits import expanding_window_folds, split_frame

if TYPE_CHECKING:  # pragma: no cover
    import optuna

logger = get_logger(__name__)


def _suggest_params(trial: optuna.Trial, seed: int) -> dict[str, Any]:
    """探索範囲の定義。

    範囲は「需要予測の日次データ・数十万行」という規模を前提に絞ってある。
    無条件に広く取ると、試行の大半が明らかに劣る領域の探索に消える。
    """
    return {
        "random_state": seed,
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 200, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 150, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
    }


def run_tuning(cfg: Config) -> dict[str, Any]:
    """ハイパーパラメータを探索し、最良のパラメータを返す。

    見つかったパラメータは自動では反映しない。``conf/config.yaml`` に
    手で書き写す運用にしている。設定ファイルが「今どのパラメータで
    動いているか」の唯一の情報源であってほしいためで、探索結果が
    黙って設定を書き換えると、差分としてレビューできなくなる。

    Raises:
        ImportError: optuna が未インストールの場合。
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - 環境依存
        msg = "optuna が未インストールです。`uv sync --extra train` を実行してください。"
        raise ImportError(msg) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    demand = read_demand_frame(cfg.paths.raw, calendar=cfg.features.calendar)
    encoder = SeriesEncoder.fit(demand)
    frame = encoder.transform(build_training_frame(demand, cfg.features))
    features = feature_columns(frame)

    folds = expanding_window_folds(
        frame.get_column("date"),
        n_splits=cfg.cv.n_splits,
        val_steps=cfg.cv.val_steps,
        gap_steps=cfg.cv.gap_steps,
    )
    splits = [split_frame(frame, fold) for fold in folds]

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, cfg.seed)
        scores: list[float] = []
        for fold_index, (train_frame, val_frame) in enumerate(splits):
            model = QuantileForecaster(quantiles=[0.5], params=params)
            model.fit(train_frame, features)
            preds = model.predict(val_frame)
            y_true = val_frame.get_column("y").to_numpy().astype(np.float64)
            scores.append(wape(y_true, preds[0.5]))
            # 明らかに劣る試行は途中で打ち切る
            trial.report(float(np.mean(scores)), step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    logger.info("探索を開始します (n_trials=%d)", cfg.tuning.n_trials)
    study.optimize(
        objective,
        n_trials=cfg.tuning.n_trials,
        timeout=cfg.tuning.timeout_seconds,
        show_progress_bar=False,
    )

    logger.info("最良 WAPE=%.4f", study.best_value)
    logger.info("最良パラメータ: %s", study.best_params)
    logger.info("conf/config.yaml の model.params に反映してから `dfc train` を実行してください。")

    history = pl.DataFrame(
        [
            {"trial": t.number, "wape": t.value, "state": str(t.state), **t.params}
            for t in study.trials
            if t.value is not None
        ]
    )
    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "history": history,
    }
