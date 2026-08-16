"""図の生成。

図は「レビューを受けるための道具」として作っている。指標の表だけでは
「どこで外しているか」が分からず、モデルの改善方針が立たない。

matplotlib は学習用の optional 依存なので、未インストールなら
分かりやすいエラーにして落とす（黙って何もしないほうが混乱を招く）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from demand_forecast.config import Config
from demand_forecast.logging_utils import get_logger

logger = get_logger(__name__)

_TOP_N_FEATURES = 20


def _require_matplotlib() -> Any:
    """matplotlib を遅延インポートし、``pyplot`` モジュールを返す。

    Raises:
        ImportError: 未インストールの場合。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # 画面のない環境でも動くようにする
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - 環境依存
        msg = "matplotlib が未インストールです。`uv sync --extra train` を実行してください。"
        raise ImportError(msg) from exc
    return plt


def plot_feature_importance(importance_csv: Path, output_path: Path) -> Path:
    """特徴量重要度の横棒グラフを描く。"""
    plt = _require_matplotlib()

    importance = pl.read_csv(importance_csv).head(_TOP_N_FEATURES).reverse()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(importance.get_column("feature").to_list(), importance.get_column("importance"))
    # 図のラベルは英語で書く。日本語フォントが入っていない環境（CI・コンテナ）で
    # 豆腐文字になるのを避けるため。
    ax.set_title(f"Top {_TOP_N_FEATURES} feature importances (split count)")
    ax.set_xlabel("importance")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("図を保存しました: %s", output_path)
    return output_path


def plot_backtest(
    predictions_path: Path,
    output_path: Path,
    *,
    horizon: int = 7,
) -> Path:
    """検証期間の実績と予測（区間つき）を系列ごとに描く。

    Args:
        predictions_path: 学習時に保存した検証予測。
        output_path: 出力先。
        horizon: 描画する予測日数。混ぜて描くと読めなくなるため1つに絞る。

    Raises:
        ValueError: 指定 horizon の予測が存在しない場合。
    """
    plt = _require_matplotlib()

    predictions = pl.read_parquet(predictions_path).filter(pl.col("feat_horizon") == horizon)
    if predictions.is_empty():
        msg = f"horizon={horizon} の検証予測がありません。"
        raise ValueError(msg)

    series_keys = (
        predictions.select(["store_id", "sku_id"]).unique().sort(["store_id", "sku_id"]).head(4)
    )
    fig, axes = plt.subplots(len(series_keys), 1, figsize=(11, 3 * len(series_keys)), sharex=True)
    axes = axes if hasattr(axes, "__len__") else [axes]

    for ax, key in zip(axes, series_keys.to_dicts(), strict=False):
        subset = predictions.filter(
            (pl.col("store_id") == key["store_id"]) & (pl.col("sku_id") == key["sku_id"])
        ).sort("date")
        dates = subset.get_column("date").to_list()
        ax.plot(dates, subset.get_column("y"), label="actual", linewidth=1.6)
        ax.plot(dates, subset.get_column("y_pred"), label="forecast (median)", linewidth=1.6)
        ax.fill_between(
            dates,
            subset.get_column("y_lower"),
            subset.get_column("y_upper"),
            alpha=0.2,
            label="prediction interval",
        )
        ax.set_title(f"{key['store_id']} / {key['sku_id']} (horizon={horizon}d)")
        ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("図を保存しました: %s", output_path)
    return output_path


def make_all_figures(cfg: Config) -> list[Path]:
    """学習後に作る図をまとめて生成する。

    Raises:
        FileNotFoundError: 学習成果物が見つからない場合。
    """
    reports_dir = Path(cfg.paths.reports_dir)
    figures_dir = reports_dir / "figures"

    importance_csv = reports_dir / "feature_importance.csv"
    predictions_path = reports_dir / "val_predictions.parquet"
    for path in (importance_csv, predictions_path):
        if not path.exists():
            msg = f"{path} がありません。先に `uv run dfc train` を実行してください。"
            raise FileNotFoundError(msg)

    return [
        plot_feature_importance(importance_csv, figures_dir / "feature_importance.png"),
        plot_backtest(predictions_path, figures_dir / "backtest.png"),
    ]
