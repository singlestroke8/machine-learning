"""着地予測の学習と評価。

評価で守っていることが3つある。

1. **時間順で検証する。** 学習期間を伸ばしながら、その先の月を検証する。
   ランダムに分けると未来で学習して過去を当てる形になり、成績が必ず良く出る。
2. **経過営業日ごとに精度を出す。** 平均だけ見ると、月末の当たりやすい
   予測に引きずられる。価値があるのは月の前半なので、そこを単独で見る。
3. **ベースラインを同じ検証データで測る。** 別々の条件で比べた改善率は
   数字の作り方の問題であって、実力ではない。
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from sales_analytics.logging_utils import get_logger
from sales_analytics.models.encoding import CategoricalEncoder
from sales_analytics.models.estimator import ForecastArtifact, QuantileForecaster
from sales_analytics.models.metrics import bias, coverage, wape
from sales_analytics.tasks.landing.baselines import BASELINES, add_baselines
from sales_analytics.tasks.landing.dataset import (
    ANCHOR_COL,
    GROUP_COL,
    MONTH_COL,
    SCALE_COL,
    STEP_COL,
    TARGET_COL,
    build_landing_frame,
    feature_columns,
)

logger = get_logger(__name__)

#: 予測区間に使う分位点
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

#: LightGBM のパラメータ。行数が3千程度なので、木は浅く小さくする。
#: 深くすると月ごとの個別事情を覚えてしまい、検証で崩れる。
MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 30,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

#: 検証に使う月数と、分割数
VAL_MONTHS = 3
N_SPLITS = 3

#: 経過営業日のうち、精度を個別に報告する位置
REPORT_STEPS: tuple[int, ...] = (1, 3, 5, 10, 15, 20)


@dataclass(frozen=True)
class FoldResult:
    """1分割ぶんの結果。"""

    index: int
    train_end: dt.date
    val_start: dt.date
    val_end: dt.date
    predictions: pl.DataFrame


def _remaining_target() -> pl.Expr:
    """目的変数を「残りいくら入るか ÷ 前年同月の着地額」にする。

    ここに至るまでに3回失敗しているので、経緯を残しておく。

    **1回目**: 着地額を円のまま予測させた。学習が成立しなかった。
    系列ごとに桁が20倍違い（営業2部 約10億 / 全社 約38億）、さらに
    水準が年々上がる（月平均 28.6億 → 46.1億）。決定木は葉の平均しか
    返せないので、**学習データの範囲を超える値を外挿できない**。

    **2回目**: 前年同月の着地額で割った比にした。まだ負けた（0.1935）。
    最終営業日を見ると理由が分かる。この時点では「累計＝着地額」で
    答えが確定しているのに、モデルの WAPE は 0.138 だった。
    **すでに確定している部分まで、木に推測させていた。**

    **3回目**: 素朴な見込み額を土台にして補正倍率を学ばせた（0.1614）。
    改善したが、まだ確定部分を掛け算で歪めていた。

    **4回目（現在）**: 確定している累計は**そのまま足す**。
    モデルには「残り営業日にいくら入るか」だけを予測させる。

        着地額 = 当月の累計（確定） ＋ 予測した残り

    最終営業日では残りがゼロなので、モデルは0を返せばよい。
    予測区間も残りの不確実性だけを表すようになり、意味がはっきりする。

    **確定している情報を、モデルに推測させない。** これが要点である。
    """
    return ((pl.col(TARGET_COL) - pl.col("cum_金額")) / pl.col(ANCHOR_COL)).alias("y")


def _month_folds(months: list[dt.date]) -> list[tuple[list[dt.date], list[dt.date]]]:
    """拡大窓の分割を、月単位で作る。

    検証期間を後ろから順に切り出し、学習期間はそのぶん短くする。
    ``gap`` は置かない。特徴量が対象月より前で打ち切られているため、
    学習と検証が重なる余地が構造的に無い。
    """
    folds: list[tuple[list[dt.date], list[dt.date]]] = []
    for split in range(N_SPLITS):
        end = len(months) - split * VAL_MONTHS
        start = end - VAL_MONTHS
        if start <= 0:
            break
        folds.append((months[:start], months[start:end]))
    return list(reversed(folds))


def _usable(frame: pl.DataFrame) -> pl.DataFrame:
    """前年同月が揃っている行だけを残す。

    データの最初の12ヶ月は、前年同月の特徴量もベースラインも作れない。
    ここを残すと**ベースラインだけが不当に悪く見える**ので、両方から外す。
    """
    return frame.filter(
        pl.col(SCALE_COL).is_not_null()
        & (pl.col(SCALE_COL) > 0)
        & pl.col(ANCHOR_COL).is_not_null()
        & (pl.col(ANCHOR_COL) > 0)
    )


def _evaluate(frame: pl.DataFrame, prediction_col: str) -> dict[str, float]:
    subset = frame.drop_nulls(prediction_col)
    if subset.height == 0:
        return {"wape": float("nan"), "bias": float("nan"), "n": 0}
    actual = subset.get_column(TARGET_COL).to_numpy()
    predicted = subset.get_column(prediction_col).to_numpy()
    return {
        "wape": wape(actual, predicted),
        "bias": bias(actual, predicted),
        "n": float(subset.height),
    }


def run_training(
    transactions: pl.DataFrame, *, reports_dir: Path, model_path: Path | None = None
) -> dict[str, Any]:
    """着地予測を学習し、評価してレポートを書き出す。"""
    frame = _usable(add_baselines(build_landing_frame(transactions)))
    features = feature_columns(frame)
    encoder = CategoricalEncoder.fit(frame, [GROUP_COL])
    frame = encoder.transform(frame)
    features = [*features, *encoder.feature_names()]

    months = sorted(frame.get_column(MONTH_COL).unique().to_list())
    folds = _month_folds(months)
    if not folds:
        msg = f"検証に使える期間がありません（{len(months)}ヶ月）。期間を延ばしてください。"
        raise ValueError(msg)

    logger.info("学習行数=%d, 特徴量数=%d, 系列数=%d", frame.height, len(features), 4)

    results: list[FoldResult] = []
    for index, (train_months, val_months) in enumerate(folds):
        train = frame.filter(pl.col(MONTH_COL).is_in(train_months))
        val = frame.filter(pl.col(MONTH_COL).is_in(val_months))

        model = QuantileForecaster(quantiles=list(QUANTILES), params=MODEL_PARAMS)
        model.fit(train.with_columns(_remaining_target()), features)
        predicted = model.predict(val)

        # 確定している累計に、予測した残りを足して円に戻す
        scale = val.get_column(ANCHOR_COL).to_numpy()
        settled = val.get_column("cum_金額").to_numpy()
        val = val.with_columns(
            pl.Series("pred_point", settled + predicted[0.5] * scale),
            pl.Series("pred_lower", settled + predicted[0.1] * scale),
            pl.Series("pred_upper", settled + predicted[0.9] * scale),
        )
        results.append(
            FoldResult(
                index=index,
                train_end=train_months[-1],
                val_start=val_months[0],
                val_end=val_months[-1],
                predictions=val,
            )
        )
        fold_wape = _evaluate(val, "pred_point")["wape"]
        logger.info(
            "fold%d: 学習 〜%s (%dヶ月) / 検証 %s〜%s  WAPE=%.4f",
            index,
            train_months[-1],
            len(train_months),
            val_months[0],
            val_months[-1],
            fold_wape,
        )

    all_predictions = pl.concat([r.predictions for r in results])
    summary = _summarize(all_predictions, results)

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "landing_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    all_predictions.write_parquet(reports_dir / "landing_predictions.parquet")

    if model_path is not None:
        final = QuantileForecaster(quantiles=list(QUANTILES), params=MODEL_PARAMS)
        final.fit(frame.with_columns(_remaining_target()), features)
        ForecastArtifact(
            model=final,
            encoder=encoder,
            metadata={
                "task": "landing",
                "trained_at": dt.datetime.now(dt.UTC).isoformat(),
                "features": features,
                "cv_wape": summary["model"]["wape"],
            },
        ).save(model_path)

    return summary


def _summarize(predictions: pl.DataFrame, folds: list[FoldResult]) -> dict[str, Any]:
    """モデルとベースラインを、同じ検証データで比較する。"""
    model = _evaluate(predictions, "pred_point")
    model["coverage"] = coverage(
        predictions.get_column(TARGET_COL).to_numpy(),
        predictions.get_column("pred_lower").to_numpy(),
        predictions.get_column("pred_upper").to_numpy(),
    )

    baselines = {b.name: _evaluate(predictions, b.column) for b in BASELINES}
    best = min(baselines.items(), key=lambda kv: kv[1]["wape"])

    by_step: list[dict[str, Any]] = []
    for step in REPORT_STEPS:
        subset = predictions.filter(pl.col(STEP_COL) == step)
        if subset.height == 0:
            continue
        row: dict[str, Any] = {"step": step, "n": subset.height}
        row["model"] = _evaluate(subset, "pred_point")["wape"]
        for b in BASELINES:
            row[b.name] = _evaluate(subset, b.column)["wape"]
        by_step.append(row)

    by_group: list[dict[str, Any]] = []
    for group in sorted(predictions.get_column(GROUP_COL).unique().to_list()):
        subset = predictions.filter(pl.col(GROUP_COL) == group)
        by_group.append(
            {
                "group": group,
                "model": _evaluate(subset, "pred_point")["wape"],
                "best_baseline": _evaluate(subset, f"base_{best[0]}")["wape"],
            }
        )

    return {
        "n_val_rows": predictions.height,
        "folds": [
            {
                "index": f.index,
                "train_end": f.train_end,
                "val_start": f.val_start,
                "val_end": f.val_end,
                "wape": _evaluate(f.predictions, "pred_point")["wape"],
            }
            for f in folds
        ],
        "model": model,
        "baselines": baselines,
        "best_baseline": {"name": best[0], **best[1]},
        "improvement_over_best": 1.0 - model["wape"] / best[1]["wape"],
        "by_step": by_step,
        "by_group": by_group,
    }


def format_summary(summary: dict[str, Any]) -> str:
    """結果を人が読める形に整える。"""
    lines: list[str] = []
    model = summary["model"]
    best = summary["best_baseline"]

    lines.append("=== 着地予測の結果（検証データ全体）===")
    lines.append(f"  モデル        WAPE {model['wape']:.4f}   bias {model['bias']:+.4f}")
    lines.append(f"  区間カバー率  {model['coverage']:.3f}（公称 0.80）")
    lines.append("")
    lines.append("=== ベースライン比較（同じ検証データ）===")
    for name, values in summary["baselines"].items():
        mark = " ← 最強" if name == best["name"] else ""
        lines.append(f"  {name:<12} WAPE {values['wape']:.4f}{mark}")
    lines.append("")
    verdict = "勝った" if summary["improvement_over_best"] > 0 else "**負けた**"
    lines.append(
        f"  最強のベースラインに対して {summary['improvement_over_best']:+.1%} → {verdict}"
    )
    lines.append("")
    lines.append("=== 経過営業日ごとの WAPE ===")
    header = f"  {'経過':>4}  {'モデル':>8}" + "".join(f"  {b.name:>10}" for b in BASELINES)
    lines.append(header)
    for row in summary["by_step"]:
        cells = "".join(f"  {row[b.name]:>10.4f}" for b in BASELINES)
        lines.append(f"  {row['step']:>3}日  {row['model']:>8.4f}{cells}")
    lines.append("")
    lines.append("=== 系列ごとの WAPE ===")
    for row in summary["by_group"]:
        lines.append(
            f"  {row['group']:<22} モデル {row['model']:.4f}   "
            f"最強ベースライン {row['best_baseline']:.4f}"
        )
    return "\n".join(lines)
