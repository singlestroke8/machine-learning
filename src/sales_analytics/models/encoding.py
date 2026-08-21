"""カテゴリ列を整数コードに変換する。

LightGBM のカテゴリ特徴量として渡すために整数化する。学習時に作った
対応表をモデルと一緒に保存し、推論時も同じ対応表を使う。
これを別々に持つと、学習と推論で違うコードが割り当てられ、
**エラーを出さないまま間違った予測を返す**ことになる。

対象の列は課題ごとに違う（部署・顧客・品名など）ので、
列名を固定せず、生成時に渡す形にしている。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

#: 未知のカテゴリに割り当てるコード。学習時に存在しなかったことを表す
UNKNOWN_CODE = -1


def encoded_name(column: str) -> str:
    """カテゴリ列に対応する特徴量の列名。"""
    return f"feat_{column}_code"


@dataclass
class CategoricalEncoder:
    """カテゴリ列 → 整数コードの対応表。

    未知のIDは ``UNKNOWN_CODE`` に落とす。落とさずに例外にすると、
    新規顧客や新商品が現れた瞬間に推論が止まってしまい、運用に耐えない。
    「知らないもの」として扱えることが要件になる。
    """

    mappings: dict[str, dict[str, int]]

    @property
    def columns(self) -> list[str]:
        return list(self.mappings)

    @classmethod
    def fit(cls, df: pl.DataFrame, columns: list[str]) -> CategoricalEncoder:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            msg = f"データに無い列を指定しています: {missing}"
            raise KeyError(msg)
        return cls(
            mappings={
                column: {
                    value: index
                    for index, value in enumerate(
                        sorted(str(v) for v in df.get_column(column).unique().to_list())
                    )
                }
                for column in columns
            }
        )

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                pl.col(column)
                .cast(pl.Utf8)
                .replace_strict(mapping, default=UNKNOWN_CODE, return_dtype=pl.Int32)
                .alias(encoded_name(column))
                for column, mapping in self.mappings.items()
            ]
        )

    def feature_names(self) -> list[str]:
        return [encoded_name(c) for c in self.mappings]

    def to_dict(self) -> dict[str, Any]:
        return {"mappings": self.mappings}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CategoricalEncoder:
        return cls(mappings={k: dict(v) for k, v in payload["mappings"].items()})


def categorical_features(columns: list[str]) -> list[str]:
    """LightGBM にカテゴリとして渡す列名を返す。

    列名の規約（``feat_<列>_code``）で判定する。対象を列挙して持つと、
    カテゴリ列を増やしたときに**追記し忘れても動いてしまい**、
    その列だけ数値として扱われていることに気づけない。
    """
    return [c for c in columns if c.startswith("feat_") and c.endswith("_code")]
