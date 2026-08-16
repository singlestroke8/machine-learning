"""Polars のスカラー値を、扱いやすい Python 型に絞り込むヘルパー。

``Series.max()`` や ``Series.item()`` の戻り値は、型の上では
「数値 or 日付 or 文字列 or None」といった広い union になる。実行時には
1つの型に決まっているのだが、型検査器から見ると何にでもなりうるため、
そのまま日付演算や比較に使うと静的検査が通らない。

「ここは日付のはず」という前提をコードに明示し、
違ったら実行時にも落とす、という形で境界を1か所にまとめている。
"""

from __future__ import annotations

import datetime as dt
from typing import Any


def as_date(value: Any) -> dt.date:
    """値を ``date`` として取り出す。

    Raises:
        TypeError: 日付として解釈できない場合。
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    msg = f"日付として扱えない値です: {value!r} (型: {type(value).__name__})"
    raise TypeError(msg)


def as_float(value: Any) -> float:
    """値を ``float`` として取り出す。

    Raises:
        TypeError: 数値として解釈できない場合。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"数値として扱えない値です: {value!r} (型: {type(value).__name__})"
        raise TypeError(msg)
    return float(value)
