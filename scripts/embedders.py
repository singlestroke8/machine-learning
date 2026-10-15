"""Voyage でテキストを 256 次元のベクトルに変換する。結果はディスクにキャッシュする。

使い方:
    make py ARGS="scripts/embedders.py"                         # voyage-3.5-lite（既定）
    make py ARGS="scripts/embedders.py --model voyage-3.5"      # Phase 3 の比較用

単体で実行すると .cache/chunks.json の全チャンクを埋め込み、件数とトークン数を表示する。
load_kb.py などからは Embedder クラスを import して使う。
"""

import argparse
import hashlib
import json
from pathlib import Path

import voyageai

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache/embeddings"
CHUNKS_PATH = ROOT / ".cache/chunks.json"

DEFAULT_MODEL = "voyage-3.5-lite"
DIM = 256  # db/init/001_schema.sql の vector(256) と一致させる
BATCH = 128  # 1 リクエストあたりのテキスト数（API 上限 1000 より十分小さく）


class Embedder:
    def __init__(self, model: str = DEFAULT_MODEL, dim: int = DIM):
        self.model = model
        self.dim = dim
        # API キーは環境変数 VOYAGE_API_KEY から読まれる（make py が .env を渡す）。
        # 429 や 5xx は SDK が指数バックオフで再試行する。
        self.client = voyageai.Client(max_retries=3, timeout=60)
        self.billed_tokens = 0  # このプロセスで実際に課金されたトークン数
        self.cache_hits = 0

    # キャッシュのキー: モデル・次元・input_type・本文のどれかが変われば別物として扱う。
    # 同じ文でも input_type が違えばベクトルが変わるため、キーに含める。
    def _cache_path(self, text: str, input_type: str) -> Path:
        raw = f"{self.model}|{self.dim}|{input_type}|{text}"
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return CACHE_DIR / f"{self.model}-{self.dim}" / f"{key}.json"

    def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        """texts を順番どおりにベクトル化して返す。input_type は "document" か "query"。"""
        if input_type not in ("document", "query"):
            raise ValueError(f"input_type は document か query: {input_type}")

        # 1. キャッシュにあるものはファイルから読み、無いものの位置だけ控える
        out: list[list[float] | None] = [None] * len(texts)
        misses: list[int] = []
        for i, t in enumerate(texts):
            p = self._cache_path(t, input_type)
            if p.exists():
                out[i] = json.loads(p.read_text())
                self.cache_hits += 1
            else:
                misses.append(i)

        # 2. 無いものだけを BATCH 件ずつ Voyage に送る（ここだけ課金される）
        for start in range(0, len(misses), BATCH):
            idx = misses[start : start + BATCH]
            res = self.client.embed(
                [texts[i] for i in idx],
                model=self.model,
                input_type=input_type,
                output_dimension=self.dim,
            )
            self.billed_tokens += res.total_tokens

            # 3. 返ってきたベクトルを元の位置に戻し、1 件ずつキャッシュへ保存する
            for i, vec in zip(idx, res.embeddings):
                if len(vec) != self.dim:
                    raise RuntimeError(f"次元が {len(vec)} で返った（期待 {self.dim}）")
                p = self._cache_path(texts[i], input_type)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(vec))
                out[i] = vec

        return out  # type: ignore[return-value]


def chunk_to_text(chunk: dict) -> str:
    """埋め込みに渡す文字列。見出しを前に付けると、本文に出てこない話題語でも引っかかる。"""
    if chunk["heading"]:
        return f"{chunk['heading']}\n{chunk['text']}"
    return chunk["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    # 1. build_kb.py が作ったチャンクを読む
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    # 2. 本文チャンクは document として埋め込む（検索時の質問側は query）
    emb = Embedder(args.model)
    vecs = emb.embed([chunk_to_text(c) for c in chunks], "document")

    # 3. 結果を表示する。2 回目以降は全件キャッシュから読むので課金 0 になる
    print(f"model: {emb.model} / dim: {emb.dim}")
    print(f"chunks: {len(vecs)} / cache hits: {emb.cache_hits} / billed tokens: {emb.billed_tokens}")
    print(f"vec[0][:5]: {[round(x, 4) for x in vecs[0][:5]]}")


if __name__ == "__main__":
    main()