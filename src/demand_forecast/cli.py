"""コマンドラインインターフェース。

パイプラインの各段階を1コマンドずつ実行できるようにしている。
一括実行のスクリプトを1本だけ用意すると、途中でこけたときに
最初からやり直すことになり、試行のサイクルが遅くなる。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from demand_forecast import __version__
from demand_forecast.config import Config, load_config
from demand_forecast.data.generate import generate_demand_data
from demand_forecast.data.loaders import read_demand_frame, validate_demand_frame, write_frame
from demand_forecast.logging_utils import configure_logging, get_logger
from demand_forecast.polars_utils import as_date, as_float

logger = get_logger(__name__)

app = typer.Typer(
    help="需要予測システムの CLI",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="設定ファイルのパス（既定: conf/config.yaml）"),
]


def _load(config_path: Path | None) -> Config:
    configure_logging()
    return load_config(config_path)


@app.command()
def version() -> None:
    """バージョンを表示する。"""
    typer.echo(f"demand-forecast {__version__}")


@app.command("generate-data")
def generate_data(
    config: ConfigOption = None,
    force: Annotated[bool, typer.Option("--force", help="既存ファイルを上書きする")] = False,
) -> None:
    """合成需要データを生成して保存する。"""
    cfg = _load(config)
    output = Path(cfg.paths.raw)

    if output.exists() and not force:
        typer.echo(f"既に存在します: {output}（上書きするには --force）")
        raise typer.Exit(code=0)

    logger.info("データを生成します: %s 〜 %s", cfg.data.start_date, cfg.data.end_date)
    df = generate_demand_data(cfg.data, seed=cfg.seed)
    validate_demand_frame(df)
    write_frame(df, output)

    summary = df.select(
        pl.len().alias("rows"),
        pl.col("units_sold").mean().round(2).alias("units_mean"),
        pl.col("units_sold").sum().alias("units_total"),
    ).to_dicts()[0]
    logger.info(
        "生成完了: %s（%d 行, 平均 %.2f 個/日, 総販売 %d 個）",
        output,
        summary["rows"],
        summary["units_mean"],
        summary["units_total"],
    )


@app.command("generate-transactions")
def generate_transactions_cmd(
    config: ConfigOption = None,
    force: Annotated[bool, typer.Option("--force", help="既存ファイルを上書きする")] = False,
) -> None:
    """B2B 取引明細（ローデータ）を生成して CSV に保存する。

    需要予測に使う日次データではなく、実務で受け取るのと同じ**集計前の明細**を作る。
    """
    from demand_forecast.data.transactions import generate_transactions, summarize

    cfg = _load(config)
    output = Path(cfg.transactions.output)

    if output.exists() and not force:
        typer.echo(f"既に存在します: {output}（上書きするには --force）")
        raise typer.Exit(code=0)

    logger.info(
        "取引明細を生成します: %s 〜 %s（顧客 %d 社）",
        cfg.transactions.start_date,
        cfg.transactions.end_date,
        cfg.transactions.n_customers,
    )
    transactions = generate_transactions(cfg.transactions, seed=cfg.seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    transactions.write_csv(output)

    typer.echo("")
    typer.echo(f"出力: {output}")
    typer.echo("=== 検算 ===")
    for key, value in summarize(transactions, cfg.transactions).items():
        typer.echo(f"  {key}: {value}")


@app.command()
def train(
    config: ConfigOption = None,
    fast: Annotated[bool, typer.Option("--fast", help="木の本数を減らして高速に回す")] = False,
    track: Annotated[bool, typer.Option("--track/--no-track", help="MLflow に記録する")] = True,
) -> None:
    """学習・評価・保存を実行する。"""
    from demand_forecast.models.train import run_training

    cfg = _load(config)
    results = run_training(cfg, fast=fast, track=track)

    summary = results["cv_summary"]
    typer.echo("")
    typer.echo("=== CV 結果 ===")
    typer.echo(f"  WAPE : {summary['wape_mean']:.4f} (±{summary['wape_std']:.4f})")
    typer.echo(f"  MAE  : {summary['mae_mean']:.3f}")
    typer.echo(f"  bias : {summary['bias_mean']:+.4f}")
    if "interval_coverage_mean" in summary:
        typer.echo(
            f"  区間カバー率: {summary['interval_coverage_mean']:.3f}"
            f" (公称 {summary.get('interval_nominal_mean', float('nan')):.2f})"
        )
    typer.echo("")
    typer.echo("=== ベースライン比較 (WAPE) ===")
    for key, value in sorted(results["baseline_summary"].items()):
        if key.endswith("_wape_mean"):
            typer.echo(f"  {key.removesuffix('_wape_mean'):<16}: {value:.4f}")
    typer.echo("")
    typer.echo(f"詳細: {Path(cfg.paths.reports_dir) / 'model_card.md'}")


@app.command()
def tune(config: ConfigOption = None) -> None:
    """ハイパーパラメータを探索する（結果は手動で config に反映する）。"""
    from demand_forecast.models.tune import run_tuning

    cfg = _load(config)
    result = run_tuning(cfg)

    output = Path(cfg.paths.reports_dir) / "tuning_history.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result["history"].write_csv(output)

    typer.echo("")
    typer.echo(f"最良 WAPE: {result['best_value']:.4f}（{result['n_trials']} 試行）")
    typer.echo("conf/config.yaml の model.params に以下を反映してください:")
    typer.echo(json.dumps(result["best_params"], indent=2))


@app.command("noise-floor")
def noise_floor(config: ConfigOption = None) -> None:
    """誤差の理論下限を推定し、直近の学習結果と突き合わせる。

    「モデルの精度が良いのか」ではなく「まだ改善の余地があるのか」を判断するための道具。
    """
    from demand_forecast.analysis import estimate_noise_floor, explain_gap

    cfg = _load(config)
    floor = estimate_noise_floor(cfg.data, seed=cfg.seed)

    typer.echo("")
    typer.echo(
        f"理論下限 WAPE (真の期待需要を知っていても避けられない誤差): {floor['oracle_wape']:.4f}"
    )

    metrics_path = Path(cfg.paths.reports_dir) / "metrics.json"
    if not metrics_path.exists():
        typer.echo("（学習結果が無いため比較は省略。先に `dfc train` を実行してください）")
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_wape = metrics["cv_summary"]["wape_mean"]
    baselines = {
        key.removesuffix("_wape_mean"): value
        for key, value in metrics["baseline_summary"].items()
        if key.endswith("_wape_mean")
    }
    best_name, best_wape = min(baselines.items(), key=lambda kv: kv[1])

    gap = explain_gap(model_wape, best_wape, floor["oracle_wape"])
    typer.echo(f"最良ベースライン ({best_name}) WAPE: {best_wape:.4f}")
    typer.echo(f"モデル WAPE: {model_wape:.4f}")
    typer.echo("")
    typer.echo(f"  ベースラインからの改善: {gap['improvement_over_baseline'] * 100:.1f}%")
    typer.echo(
        f"  学習可能な余地のうち回収した割合: {gap['captured_share_of_learnable'] * 100:.1f}%"
    )
    typer.echo(f"  理論下限までの残り: {gap['remaining_gap']:.4f}")


@app.command()
def figures(config: ConfigOption = None) -> None:
    """学習結果から図を生成する。"""
    from demand_forecast.reporting import make_all_figures

    cfg = _load(config)
    for path in make_all_figures(cfg):
        typer.echo(f"生成: {path}")


@app.command()
def forecast(
    config: ConfigOption = None,
    store: Annotated[str, typer.Option(help="店舗ID")] = "S01",
    sku: Annotated[str, typer.Option(help="商品ID")] = "SKU01",
    days: Annotated[int, typer.Option(help="何日先まで予測するか")] = 14,
) -> None:
    """保存済みモデルで、データ末尾を origin とした予測を出力する。

    API を立てずに動作確認するための入口。実データの末尾を origin として、
    そこから先の価格・販促は「直近の水準が続く」と仮定して埋める。
    """
    from demand_forecast.models.estimator import ForecastArtifact
    from demand_forecast.models.predict import forecast as run_forecast

    cfg = _load(config)
    artifact = ForecastArtifact.load(cfg.api.model_path)

    demand = read_demand_frame(cfg.paths.raw)
    history = demand.filter((pl.col("store_id") == store) & (pl.col("sku_id") == sku))
    if history.is_empty():
        typer.echo(f"該当する系列がありません: store={store}, sku={sku}")
        raise typer.Exit(code=1)

    origin = as_date(history.get_column("date").max())
    last_price = as_float(history.sort("date").get_column("price").tail(1).item())
    horizon = min(days, artifact.feature_config.horizon)

    future = pl.DataFrame(
        {
            "date": [origin + dt.timedelta(days=h) for h in range(1, horizon + 1)],
            "store_id": [store] * horizon,
            "sku_id": [sku] * horizon,
            "price": [last_price] * horizon,
            "promo_flag": [0] * horizon,
        },
        schema={
            "date": pl.Date,
            "store_id": pl.Utf8,
            "sku_id": pl.Utf8,
            "price": pl.Float64,
            "promo_flag": pl.Int8,
        },
    )

    result = run_forecast(artifact, history, future)
    typer.echo(f"origin={origin} / store={store} / sku={sku}")
    typer.echo(f"{'date':<12}{'h':>3}{'point':>9}{'lower':>9}{'upper':>9}")
    for row in result.to_dicts():
        date_label = str(row["date"])
        typer.echo(
            f"{date_label:<12}{row['horizon']:>3}"
            f"{row['point']:>9.1f}{row['lower']:>9.1f}{row['upper']:>9.1f}"
        )


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="待ち受けアドレス")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="待ち受けポート")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="変更を検知して再起動する")] = False,
) -> None:
    """推論APIを起動する。"""
    import uvicorn

    configure_logging()
    uvicorn.run("demand_forecast.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":  # pragma: no cover
    app()
