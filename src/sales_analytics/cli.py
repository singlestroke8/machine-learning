"""コマンドラインインターフェース。

パイプラインの各段階を1コマンドずつ実行できるようにしている。
一括実行のスクリプトを1本だけ用意すると、途中でこけたときに
最初からやり直すことになり、試行のサイクルが遅くなる。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from sales_analytics import __version__
from sales_analytics.config import Config, load_config
from sales_analytics.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

app = typer.Typer(
    help="販売管理データを題材にした機械学習の学習記録",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="設定ファイルのパス（既定: conf/config.yaml）"),
]


def _load(config_path: Path | None) -> Config:
    configure_logging()
    cfg = load_config(config_path)
    logger.debug("設定を読み込みました: %s", config_path or "conf/config.yaml")
    return cfg


@app.command()
def version() -> None:
    """バージョンを表示する。"""
    typer.echo(f"sales-analytics {__version__}")


@app.command("generate")
def generate_data(
    config: ConfigOption = None,
    force: Annotated[bool, typer.Option("--force", help="既存ファイルを上書きする")] = False,
) -> None:
    """取引明細・顧客マスタ・異常ラベルを生成する。

    生成後、そのまま合格基準の検査まで行う。
    「生成できた」と「使えるデータができた」は別なので、必ず両方を通す。
    """
    from sales_analytics.data.generator import generate
    from sales_analytics.data.validate import format_checks, run_checks

    cfg = _load(config)
    raw_dir = cfg.paths.raw_dir
    targets = {
        "transactions": raw_dir / "transactions.csv",
        "customers": raw_dir / "customers.csv",
        "anomaly_labels": raw_dir / "anomaly_labels.csv",
    }
    existing = [p for p in targets.values() if p.exists()]
    if existing and not force:
        typer.echo(f"既に存在します: {existing[0]}（上書きするには --force）")
        raise typer.Exit(code=1)

    logger.info("取引明細を生成します（顧客 %d 社）", cfg.transactions.n_customers)
    data = generate(cfg.transactions, seed=cfg.seed)

    raw_dir.mkdir(parents=True, exist_ok=True)
    data.transactions.write_csv(targets["transactions"])
    data.customers.write_csv(targets["customers"])
    data.anomaly_labels.write_csv(targets["anomaly_labels"])
    for name, path in targets.items():
        typer.echo(f"書き出しました: {path}（{name}）")

    typer.echo("")
    typer.echo("=== 生成結果 ===")
    typer.echo(f"  明細行数     : {data.transactions.height:,}")
    typer.echo(f"  顧客数       : {data.customers.height}")
    typer.echo(f"  異常ラベル   : {data.anomaly_labels.height} 行")
    typer.echo("")
    typer.echo("=== 合格基準の検査 ===")
    checks = run_checks(data.transactions, data.anomaly_labels)
    typer.echo(format_checks(checks))
    failed = [c for c in checks if not c.passed]
    if failed:
        typer.echo("")
        typer.echo(f"{len(failed)} 件が基準を満たしていません。生成器の調整が必要です。")
        raise typer.Exit(code=1)


@app.command("check-data")
def check_data(config: ConfigOption = None) -> None:
    """生成済みのデータが合格基準を満たしているかだけを検査する。"""
    import polars as pl

    from sales_analytics.data.validate import format_checks, run_checks

    cfg = _load(config)
    transactions_path = cfg.paths.raw_dir / "transactions.csv"
    labels_path = cfg.paths.raw_dir / "anomaly_labels.csv"
    if not transactions_path.exists():
        typer.echo(f"{transactions_path} がありません。先に `sales generate` を実行してください。")
        raise typer.Exit(code=1)

    transactions = pl.read_csv(transactions_path, try_parse_dates=True)
    labels = pl.read_csv(labels_path, try_parse_dates=True)
    checks = run_checks(transactions, labels)
    typer.echo(format_checks(checks))
    if any(not c.passed for c in checks):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
