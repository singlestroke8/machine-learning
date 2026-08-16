"""学習パイプライン。

流れは次のとおり。

1. 需要データを読み込み、特徴量行列を組み立てる
2. 拡大窓の時系列CVで、モデルとベースラインを**同じ検証期間で**評価する
3. 全期間で最終モデルを学習し直す
4. モデル・指標・特徴量重要度・モデルカードを保存する

CV は「本番で出る精度の見積り」、最終モデルは「実際に配るもの」であり、
役割が違う。CV に使ったモデルをそのまま配ると、最新のデータを
学習に使えていない分だけ損をする。
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from demand_forecast.models.baselines import compute_baselines
from demand_forecast.models.estimator import ForecastArtifact, QuantileForecaster
from demand_forecast.models.metrics import evaluate_quantile_forecast, wape
from demand_forecast.models.splits import expanding_window_folds, split_frame

logger = get_logger(__name__)

# fast モード（CI のスモークテスト用）で上書きするパラメータ
_FAST_OVERRIDES: dict[str, Any] = {"n_estimators": 60, "num_leaves": 15}


def _summarize_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """fold ごとの指標を平均・標準偏差にまとめる。"""
    if not fold_metrics:
        return {}
    keys = sorted(fold_metrics[0])
    summary: dict[str, float] = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics if key in m], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    return summary


def _horizon_breakdown(val: pl.DataFrame, y_pred: np.ndarray) -> pl.DataFrame:
    """horizon 別の WAPE を計算する。

    「14日先まで予測できます」と言うとき、1日先と14日先の精度は当然違う。
    どこまでなら実用に耐えるかを示すために必ず分解して出す。
    """
    scored = val.select(["feat_horizon", "y"]).with_columns(
        pl.Series("y_pred", y_pred, dtype=pl.Float64)
    )
    return (
        scored.group_by("feat_horizon")
        .agg(
            (pl.col("y") - pl.col("y_pred")).abs().sum().alias("_abs_err"),
            pl.col("y").abs().sum().alias("_abs_actual"),
            pl.len().alias("n_rows"),
        )
        .with_columns((pl.col("_abs_err") / pl.col("_abs_actual")).alias("wape"))
        .drop("_abs_err", "_abs_actual")
        .sort("feat_horizon")
        .rename({"feat_horizon": "horizon"})
    )


def _write_model_card(
    path: Path,
    *,
    cfg: Config,
    summary: dict[str, float],
    baseline_summary: dict[str, float],
    horizon_table: pl.DataFrame,
    n_rows: int,
    n_series: int,
    trained_at: str,
) -> None:
    """モデルカード（何を学習し、どこまで信用してよいかの説明書）を書き出す。"""
    improvement = ""
    model_wape = summary.get("wape_mean")
    best_baseline = min(
        ((k, v) for k, v in baseline_summary.items() if k.endswith("_wape_mean")),
        key=lambda kv: kv[1],
        default=None,
    )
    if model_wape is not None and best_baseline is not None:
        name = best_baseline[0].removesuffix("_wape_mean")
        delta = (best_baseline[1] - model_wape) / best_baseline[1] * 100
        improvement = (
            f"最良のベースライン（{name}, WAPE {best_baseline[1]:.4f}）に対して "
            f"**{delta:.1f}% の改善**。\n"
        )

    horizon_lines = "\n".join(
        f"| {row['horizon']} | {row['wape']:.4f} | {row['n_rows']:,} |"
        for row in horizon_table.to_dicts()
    )

    metric_lines = "\n".join(
        f"| {k.removesuffix('_mean')} | {v:.4f} | {summary.get(k.replace('_mean', '_std'), 0):.4f} |"
        for k, v in sorted(summary.items())
        if k.endswith("_mean")
    )

    content = f"""# モデルカード: 需要予測モデル

自動生成ファイル（`uv run dfc train` で更新される）。手で編集しない。

## 概要

| 項目 | 値 |
| --- | --- |
| 学習日時 | {trained_at} |
| アルゴリズム | LightGBM 分位点回帰 (q={cfg.model.quantiles}) |
| 予測対象 | 店舗×商品の日次販売数量 |
| 予測範囲 | 1〜{cfg.features.horizon} 日先 |
| 学習行数 | {n_rows:,} |
| 系列数 | {n_series} |
| 検証方法 | 拡大窓 時系列CV（{cfg.cv.n_splits} 分割 × {cfg.cv.val_days} 日） |

## 検証結果（CV 平均）

{improvement}
| 指標 | 平均 | 標準偏差 |
| --- | --- | --- |
{metric_lines}

## ベースラインとの比較（WAPE, 小さいほど良い）

| 手法 | WAPE |
| --- | --- |
| **本モデル** | **{summary.get("wape_mean", float("nan")):.4f}** |
""" + "\n".join(
        f"| {k.removesuffix('_wape_mean')} | {v:.4f} |"
        for k, v in sorted(baseline_summary.items())
        if k.endswith("_wape_mean")
    ) + f"""

## horizon 別の精度

| 予測日数 | WAPE | 評価行数 |
| --- | --- | --- |
{horizon_lines}

## 使用上の注意

- 学習データは合成データである。実データに差し替えた場合、ここに載っている
  数字はいずれも意味を持たない（必ず再学習・再評価すること）。
- 価格と販促は「予測時点で計画が確定している」前提でモデルに与えている。
  価格が事前に確定しない業務には、このままでは適用できない。
- 欠品による販売数量の打ち切りをモデル化していない。実データでは
  「売れなかった」と「在庫が無かった」の区別が必要になる。
- 学習期間から大きく離れた将来（構造変化後）には外挿できない。
  定期的な再学習を前提とする。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("モデルカードを書き出しました: %s", path)


def run_training(cfg: Config, *, fast: bool = False, track: bool = True) -> dict[str, Any]:
    """学習・評価・保存を一気通貫で実行する。

    Args:
        cfg: 設定。
        fast: True なら木の本数を減らして高速に回す（CI のスモーク用）。
        track: True なら MLflow に記録する（未インストールなら自動でスキップ）。

    Returns:
        CV の要約指標などを含む辞書。
    """
    trained_at = dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds")
    reports_dir = Path(cfg.paths.reports_dir)

    logger.info("データを読み込みます: %s", cfg.paths.raw)
    demand = read_demand_frame(cfg.paths.raw)

    logger.info("特徴量を生成します (horizon=1〜%d)", cfg.features.horizon)
    encoder = SeriesEncoder.fit(demand)
    frame = encoder.transform(build_training_frame(demand, cfg.features))
    features = feature_columns(frame)
    n_series = demand.select(["store_id", "sku_id"]).unique().height
    logger.info("学習行数=%d, 特徴量数=%d, 系列数=%d", frame.height, len(features), n_series)

    params = dict(cfg.model.params)
    params["random_state"] = cfg.seed
    if fast:
        params.update(_FAST_OVERRIDES)
        logger.warning("fast モードで実行しています。精度は本来より低く出ます。")

    n_splits = 2 if fast else cfg.cv.n_splits
    folds = expanding_window_folds(
        frame.get_column("date"),
        n_splits=n_splits,
        val_days=cfg.cv.val_days,
        gap_days=cfg.cv.gap_days,
    )

    short_window = min(cfg.features.rolling_windows)
    fold_metrics: list[dict[str, float]] = []
    baseline_fold_metrics: list[dict[str, float]] = []
    horizon_tables: list[pl.DataFrame] = []

    for fold in folds:
        logger.info(fold.describe())
        train_frame, val_frame = split_frame(frame, fold)
        if train_frame.is_empty() or val_frame.is_empty():
            msg = f"{fold.describe()} で学習または検証データが空になりました。"
            raise ValueError(msg)

        model = QuantileForecaster(quantiles=cfg.model.quantiles, params=params)
        model.fit(train_frame, features)
        preds = model.predict(val_frame)

        y_true = val_frame.get_column("y").to_numpy().astype(np.float64)
        metrics = evaluate_quantile_forecast(y_true, dict(preds))
        fold_metrics.append(metrics)
        logger.info(
            "  -> WAPE=%.4f  MAE=%.3f  カバー率=%.3f",
            metrics["wape"],
            metrics["mae"],
            metrics.get("interval_coverage", float("nan")),
        )

        baselines = compute_baselines(val_frame, window=short_window)
        baseline_fold_metrics.append(
            {f"{name}_wape": wape(y_true, pred) for name, pred in baselines.items()}
        )

        horizon_tables.append(
            _horizon_breakdown(val_frame, preds[0.5]).with_columns(
                pl.lit(fold.index, dtype=pl.Int32).alias("fold")
            )
        )

        # 最新 fold の予測は、あとから目で確認できるよう保存しておく。
        # 指標だけ見ていると「平均は良いが特定系列で外し続けている」に気づけない。
        if fold.index == folds[-1].index:
            quantile_keys = sorted(preds)
            reports_dir.mkdir(parents=True, exist_ok=True)
            val_frame.select(["date", "store_id", "sku_id", "feat_horizon", "y"]).with_columns(
                pl.Series("y_pred", preds[0.5], dtype=pl.Float64),
                pl.Series("y_lower", preds[quantile_keys[0]], dtype=pl.Float64),
                pl.Series("y_upper", preds[quantile_keys[-1]], dtype=pl.Float64),
            ).write_parquet(reports_dir / "val_predictions.parquet")

    summary = _summarize_folds(fold_metrics)
    baseline_summary = _summarize_folds(baseline_fold_metrics)

    horizon_table = (
        pl.concat(horizon_tables)
        .group_by("horizon")
        .agg(
            (pl.col("wape") * pl.col("n_rows")).sum().alias("_weighted"),
            pl.col("n_rows").sum().alias("n_rows"),
        )
        .with_columns((pl.col("_weighted") / pl.col("n_rows")).alias("wape"))
        .drop("_weighted")
        .sort("horizon")
    )

    logger.info("全期間で最終モデルを学習します")
    final_model = QuantileForecaster(quantiles=cfg.model.quantiles, params=params)
    final_model.fit(frame, features)

    importance = final_model.feature_importance()
    artifact = ForecastArtifact(
        model=final_model,
        encoder=encoder,
        feature_config=cfg.features,
        metadata={
            "trained_at": trained_at,
            "n_train_rows": frame.height,
            "n_series": n_series,
            "n_features": len(features),
            "data_start": str(demand.get_column("date").min()),
            "data_end": str(demand.get_column("date").max()),
            "cv_summary": summary,
            "baseline_summary": baseline_summary,
            "params": params,
            "fast_mode": fast,
        },
    )
    artifact.save(Path(cfg.paths.model_dir) / "model.joblib")

    reports_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "trained_at": trained_at,
        "fast_mode": fast,
        "n_train_rows": frame.height,
        "n_features": len(features),
        "n_series": n_series,
        "folds": [asdict(f) | {"_repr": f.describe()} for f in folds],
        "cv_summary": summary,
        "baseline_summary": baseline_summary,
        "horizon_wape": horizon_table.to_dicts(),
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    importance.write_csv(reports_dir / "feature_importance.csv")
    _write_model_card(
        reports_dir / "model_card.md",
        cfg=cfg,
        summary=summary,
        baseline_summary=baseline_summary,
        horizon_table=horizon_table,
        n_rows=frame.height,
        n_series=n_series,
        trained_at=trained_at,
    )

    if track:
        # 実験記録は補助機能である。トラッキング先の不調で、
        # 成功した学習の成果物まで失われるのは割に合わない。
        try:
            _log_to_mlflow(cfg, results)
        except Exception:
            logger.exception("MLflow への記録に失敗しました（学習結果は保存済みです）")

    return results


def _log_to_mlflow(cfg: Config, results: dict[str, Any]) -> None:
    """MLflow に実験結果を記録する（未インストールならスキップ）。

    実験管理は「あとから自分の判断を検証できるようにする」ためのもので、
    ここが無くても学習は完走できるべきである。そのため optional 依存にし、
    未インストール時は警告だけ出して先に進む。
    """
    try:
        import mlflow
    except ImportError:
        logger.warning(
            "mlflow が未インストールのため実験記録をスキップします"
            "（`uv sync --extra train` で導入できます）"
        )
        return

    mlflow.set_tracking_uri(cfg.paths.mlflow_tracking_uri)
    mlflow.set_experiment("demand-forecast")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "horizon": cfg.features.horizon,
                "quantiles": cfg.model.quantiles,
                "n_splits": cfg.cv.n_splits,
                "val_days": cfg.cv.val_days,
                "n_features": results["n_features"],
                "fast_mode": results["fast_mode"],
                **{f"lgbm_{k}": v for k, v in cfg.model.params.items()},
            }
        )
        mlflow.log_metrics(
            {
                k: v
                for k, v in {**results["cv_summary"], **results["baseline_summary"]}.items()
                if np.isfinite(v)
            }
        )
        reports_dir = Path(cfg.paths.reports_dir)
        for name in ("metrics.json", "feature_importance.csv", "model_card.md"):
            path = reports_dir / name
            if path.exists():
                mlflow.log_artifact(str(path))
        logger.info("MLflow に記録しました (tracking_uri=%s)", cfg.paths.mlflow_tracking_uri)
