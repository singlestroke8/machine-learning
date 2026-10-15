"""chunks テーブルの 1 モデル分を kb.json に書き出す（Phase 5 で Lambda に同梱する）。

使い方:
    make py ARGS="scripts/export_kb.py"                       # voyage-3.5-lite（既定）
    make py ARGS="scripts/export_kb.py --model voyage-3.5"

Lambda からはローカルの Postgres に届かないため、検索に必要なものをこの 1 ファイルにまとめ、
Lambda 側では全件とのコサイン類似度を計算する（数百件なら DB なしで十分速い）。
DB を正とし、そこから書き出す。ローカル検索（src/db.ts）と Lambda で同じ中身を使うため。
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedders import DEFAULT_MODEL, DIM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "dist/kb.json"  # dist/ は .gitignore 済み

ROUND = 4  # 小数 4 桁に丸める。類似度の順位はほぼ変わらず、ファイルが半分以下になる


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    # 1. DB から指定モデルの行を id 順に読む
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        # embedding::text は "[0.01,-0.02,...]" という文字列で返り、そのまま JSON として読める。
        # pgvector ライブラリの版によって戻り値の型（numpy 配列 / Vector）が違うのを避けるため
        rows = conn.execute(
            "SELECT id, page_key, title, heading, text, embedding::text"
            " FROM chunks WHERE model = %s ORDER BY id",
            (args.model,),
        ).fetchall()

    # 2. 0 件なら load_kb.py の実行忘れなので止める（空の kb.json を Lambda に載せない）
    if not rows:
        raise SystemExit(f"chunks に model={args.model} の行がありません。先に load_kb.py を実行してください")

    # 3. JSON にできる形へ変換する。ベクトルは文字列 → float のリスト → 小数 4 桁に丸める
    chunks = [
        {
            "id": id_,
            "page_key": page_key,
            "title": title,
            "heading": heading or "",
            "text": text,
            "vec": [round(x, ROUND) for x in json.loads(emb)],
        }
        for id_, page_key, title, heading, text, emb in rows
    ]

    # 4. メタ情報を付けて書き出す。Lambda 側は model と dim を見て、質問の埋め込み条件を合わせる
    kb = {
        "version": 1,
        "model": args.model,
        "dim": DIM,
        "input_type": "document",  # 質問側は query で埋め込む（非対称）
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunks": chunks,
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    # separators で空白を削り、サイズを小さくする
    OUT_PATH.write_text(json.dumps(kb, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # 5. 件数とサイズを表示する（Lambda のパッケージ上限 50MB に対して十分小さいかの確認）
    print(f"model: {args.model} / chunks: {len(chunks)}")
    print(f"-> {OUT_PATH.relative_to(ROOT)} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()