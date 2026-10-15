"""埋め込みプロバイダ。

- VoyageEmbedder: 本番用（Voyage AI API）。VOYAGE_API_KEY が必要。
- MockEmbedder:   開発・CI用。文字n-gramのハッシュ埋め込み（決定的・API不要）。
                  語彙一致ベースラインとしてパイプライン検証と閾値の当たり付けに使う。
                  意味的な言い換え耐性は測れないため、最終的な閾値決定は
                  Voyage で再実行すること。

埋め込みは rag/.cache/ にキャッシュし、同一テキストの再課金を防ぐ。
"""

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from textmatch import normalize

CACHE_DIR = Path(__file__).resolve().parent / ".cache"


class _CachedEmbedder:
    provider = "base"
    model = ""
    dim = 0

    def _key(self, text: str, input_type: str) -> str:
        raw = f"{self.provider}|{self.model}|{self.dim}|{input_type}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / self.provider / f"{key}.json"

    def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        """texts をベクトル化して返す。input_type は 'query' か 'document'。"""
        out: list[list[float] | None] = [None] * len(texts)
        misses: list[int] = []
        for i, t in enumerate(texts):
            p = self._cache_path(self._key(t, input_type))
            if p.exists():
                out[i] = json.loads(p.read_text())
            else:
                misses.append(i)
        if misses:
            vecs = self._embed_batch([texts[i] for i in misses], input_type)
            for i, v in zip(misses, vecs):
                p = self._cache_path(self._key(texts[i], input_type))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(v))
                out[i] = v
        return out  # type: ignore[return-value]

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        raise NotImplementedError


class MockEmbedder(_CachedEmbedder):
    """文字2〜3gramをハッシュでdim次元に畳み込むTFベクトル（L2正規化）。
    1-gramは日本語テキスト同士を無差別に似せてしまうため使わない。"""

    provider = "mock"
    model = "char-ngram-hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        vecs = []
        for text in texts:
            t = normalize(text)
            v = [0.0] * self.dim
            for n in (2, 3):
                for i in range(max(0, len(t) - n + 1)):
                    g = t[i : i + n]
                    h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
                    v[h % self.dim] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vecs.append([x / norm for x in v])
        return vecs


class VoyageEmbedder(_CachedEmbedder):
    provider = "voyage"
    ENDPOINT = "https://api.voyageai.com/v1/embeddings"
    BATCH = 100

    def __init__(self, model: str = "voyage-3.5-lite", dim: int = 256):
        key = os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise SystemExit("VOYAGE_API_KEY が未設定です（--provider mock で代替可）")
        self.api_key = key
        self.model = model
        self.dim = dim

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            chunk = texts[i : i + self.BATCH]
            body = json.dumps({
                "input": chunk,
                "model": self.model,
                "input_type": input_type,
                "output_dimension": self.dim,
            }).encode("utf-8")
            req = urllib.request.Request(
                self.ENDPOINT,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=60) as res:
                        data = json.loads(res.read().decode("utf-8"))
                    break
                except urllib.error.HTTPError as e:
                    if e.code in (429, 500, 502, 503) and attempt < 3:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    raise
            items = sorted(data["data"], key=lambda d: d["index"])
            vecs.extend(d["embedding"] for d in items)
        return vecs


def make_embedder(provider: str, model: str | None, dim: int):
    if provider == "voyage":
        return VoyageEmbedder(model or "voyage-3.5-lite", dim)
    if provider == "mock":
        return MockEmbedder(dim)
    raise SystemExit(f"未知のprovider: {provider}")
