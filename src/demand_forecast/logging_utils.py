"""ロギングの共通設定。

ライブラリ側のモジュールでは ``logging.basicConfig`` を呼ばず、
エントリポイント（CLI・API）で一度だけ ``configure_logging`` を呼ぶ。
モジュールが勝手にルートロガーを設定すると、
呼び出し側のログ設計を壊してしまうため。
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | int | None = None) -> None:
    """プロセス全体のロギングを設定する（複数回呼んでも安全）。

    Args:
        level: ログレベル。``None`` の場合は環境変数 ``DFC_LOG_LEVEL``、
            それも無ければ ``INFO``。
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = level if level is not None else os.environ.get("DFC_LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.setLevel(resolved)
    root.handlers = [handler]
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """モジュール用のロガーを取得する。"""
    return logging.getLogger(name)
