"""LightGBM による分位点需要予測モデル。

点予測ひとつでは在庫の意思決定ができない。「明日 100 個売れる見込み」と
言われても、何個仕入れるべきかは需要のばらつき次第で変わるためである。
そこで q=0.1 / 0.5 / 0.9 の3本を独立に学習し、点予測（q=0.5）と
予測区間を同時に返す。詳細は ``docs/adr/0007-quantile-regression.md``。

独立学習に伴う分位点交差（q=0.1 の予測が q=0.9 を上回る）は、
予測後にソートして解消している。単純だが、実務ではこの後処理で十分機能する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from lightgbm import LGBMRegressor

from sales_analytics.logging_utils import get_logger
from sales_analytics.models.encoding import CategoricalEncoder, categorical_features

logger = get_logger(__name__)

ARTIFACT_FORMAT_VERSION = 1


def _to_matrix(frame: pl.DataFrame, feature_names: list[str]) -> np.ndarray:
    """特徴量行列を float64 の numpy 配列に変換する。

    Raises:
        KeyError: 学習時の特徴量がフレームに存在しない場合。
    """
    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        msg = f"推論に必要な特徴量が不足しています: {missing}"
        raise KeyError(msg)
    return frame.select(feature_names).to_numpy().astype(np.float64)


@dataclass
class QuantileForecaster:
    """分位点ごとの LightGBM をまとめて扱うモデル。"""

    quantiles: list[float]
    params: dict[str, Any]
    feature_names: list[str] = field(default_factory=list)
    models: dict[float, LGBMRegressor] = field(default_factory=dict)

    @property
    def point_quantile(self) -> float:
        """点予測に用いる分位点。"""
        return 0.5

    def fit(self, frame: pl.DataFrame, feature_names: list[str]) -> QuantileForecaster:
        """分位点ごとにモデルを学習する。

        Args:
            frame: ``y`` カラムを含む特徴量行列。
            feature_names: 特徴量として使うカラム名（順序が保存される）。
        """
        self.feature_names = list(feature_names)
        x = _to_matrix(frame, self.feature_names)
        y = frame.get_column("y").to_numpy().astype(np.float64)

        cat_names = categorical_features(self.feature_names)
        cat_indices = [self.feature_names.index(c) for c in cat_names]

        self.models = {}
        for q in self.quantiles:
            model = LGBMRegressor(objective="quantile", alpha=q, verbose=-1, **self.params)
            model.fit(
                x,
                y,
                categorical_feature=cat_indices if cat_indices else "auto",
            )
            self.models[q] = model
            logger.debug("分位点 q=%s のモデルを学習しました", q)
        return self

    def predict(self, frame: pl.DataFrame) -> dict[float, np.ndarray]:
        """分位点ごとの予測を返す。

        需要は負にならないため 0 で下限を切り、分位点の大小関係が
        逆転している行はソートして整合させる。

        Raises:
            RuntimeError: 未学習の状態で呼ばれた場合。
        """
        if not self.models:
            msg = "モデルが未学習です。先に fit() を呼んでください。"
            raise RuntimeError(msg)

        x = _to_matrix(frame, self.feature_names)
        ordered = sorted(self.models)
        raw = np.column_stack([self.models[q].predict(x) for q in ordered])
        raw = np.maximum(raw, 0.0)
        # 分位点交差の解消（行ごとに昇順へ並べ替える）
        raw.sort(axis=1)
        return {q: raw[:, i] for i, q in enumerate(ordered)}

    def feature_importance(self, quantile: float | None = None) -> pl.DataFrame:
        """特徴量重要度（分割回数）を降順で返す。

        Raises:
            KeyError: 指定した分位点のモデルが存在しない場合。
        """
        q = self.point_quantile if quantile is None else quantile
        if q not in self.models:
            msg = f"分位点 {q} のモデルがありません（保持しているのは {sorted(self.models)}）"
            raise KeyError(msg)
        return pl.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.models[q].feature_importances_.astype(np.int64),
            }
        ).sort("importance", descending=True)


@dataclass
class ForecastArtifact:
    """モデルと、それを再現・運用するのに必要な一切をまとめた保存単位。

    モデル単体を保存すると、推論時に「学習時とどの特徴量設定だったか」が
    分からなくなり、静かに間違った前処理で推論する事故につながる。
    ID対応表と学習メタデータを同梱して1ファイルにしている。

    ``metadata`` に何を入れるかは課題ごとに違う。特徴量の設定・学習日時・
    CV の結果など、**推論時に「学習時と同じ前処理か」を確認できるもの**を入れる。
    """

    model: QuantileForecaster
    encoder: CategoricalEncoder
    metadata: dict[str, Any] = field(default_factory=dict)
    format_version: int = ARTIFACT_FORMAT_VERSION

    def save(self, path: Path | str) -> Path:
        """joblib 形式で保存する。"""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        logger.info("モデルを保存しました: %s", file_path)
        return file_path

    @staticmethod
    def load(path: Path | str) -> ForecastArtifact:
        """保存済みモデルを読み込む。

        Raises:
            FileNotFoundError: ファイルが存在しない場合。
            ValueError: 保存形式のバージョンが合わない場合。
        """
        file_path = Path(path)
        if not file_path.exists():
            msg = f"モデルファイルが見つかりません: {file_path}\n先に学習を実行してください。"
            raise FileNotFoundError(msg)

        artifact = joblib.load(file_path)
        if not isinstance(artifact, ForecastArtifact):
            msg = f"想定外の形式のファイルです: {type(artifact)!r}"
            raise ValueError(msg)
        if artifact.format_version != ARTIFACT_FORMAT_VERSION:
            msg = (
                f"モデルの保存形式のバージョンが一致しません "
                f"(ファイル={artifact.format_version}, 実行中={ARTIFACT_FORMAT_VERSION})。"
                " 再学習してください。"
            )
            raise ValueError(msg)
        return artifact
