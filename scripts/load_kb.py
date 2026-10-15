"""チャンクと埋め込みを Postgres の chunks テーブルに投入する。

使い方:
    make py ARGS="scripts/load_kb.py"                         # voyage-3.5-lite（既定）
    make py ARGS="scripts/load_kb.py --model voyage-3.5"      # Phase 3 の比較用

前提: build_kb.py で .cache/chunks.json を作ってあること。
埋め込みは embedders.py のキャッシュを使うので、embedders.py 実行済みなら課金は 0。
同じモデルの行は毎回すべて消して入れ直す（チャンク id は通し番号でずれうるため、差分更新はしない）。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

# python -I（Dockerfile の ENTRYPOINT）はスクリプトのフォルダを import 先に入れないため、明示的に足す
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedders import CHUNKS_PATH, DEFAULT_MODEL, Embedder, chunk_to_text  # noqa: E402

INSERT_SQL = """
    INSERT INTO chunks (page_key, title, heading, text, model, embedding)
    VALUES (%s, %s, %s, %s, %s, %s)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    # 1. build_kb.py が作ったチャンクを読む
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    # 2. 埋め込みを用意する（キャッシュにあればファイルから、無ければ Voyage から）
    emb = Embedder(args.model)
    vecs = emb.embed([chunk_to_text(c) for c in chunks], "document")

    # 3. DB に接続する。DATABASE_URL は docker-compose.yml が py コンテナに渡している
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        # vector 型と numpy 配列を相互変換できるようにする
        register_vector(conn)

        # 4. 削除と投入を 1 つのトランザクションにまとめる。途中で失敗したら丸ごと元に戻る
        with conn.transaction():
            deleted = conn.execute("DELETE FROM chunks WHERE model = %s", (emb.model,)).rowcount
            with conn.cursor() as cur:
                cur.executemany(INSERT_SQL, [
                    (c["page_key"], c["title"], c["heading"] or None, c["text"], emb.model, np.array(v))
                    for c, v in zip(chunks, vecs)
                ])

        # 5. 投入後の件数をモデル別に数えて表示する
        rows = conn.execute("SELECT model, count(*) FROM chunks GROUP BY model ORDER BY model").fetchall()

    print(f"model: {emb.model} / deleted: {deleted} / inserted: {len(chunks)}")
    print(f"billed tokens: {emb.billed_tokens} / cache hits: {emb.cache_hits}")
    for model, n in rows:
        print(f"  chunks[{model}] = {n}")


if __name__ == "__main__":
    main()