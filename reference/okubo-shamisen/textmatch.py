"""Tier0（文章ファジーマッチ）の参照実装。

ここの normalize / bigram_cosine が仕様の単一の真実。
ウィジェット（chat-widget.js）とサーバ（PHP）に移植する際は
必ずこのファイルと挙動を一致させること（eval.py が回帰テストになる）。
"""

import re
import unicodedata

_STRIP = re.compile(r"[\s。、．，・！!？?「」『』()（）\[\]〜~\-–—/／…•*※:：;；\"'’”]+")


def normalize(text: str) -> str:
    """NFKC正規化 → 小文字化 → 空白・記号除去。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return _STRIP.sub("", text)


def bigrams(text: str) -> dict[str, int]:
    """正規化済みテキストの文字バイグラム頻度。1文字の場合はその文字自身。"""
    if len(text) < 2:
        return {text: 1} if text else {}
    grams: dict[str, int] = {}
    for i in range(len(text) - 1):
        g = text[i : i + 2]
        grams[g] = grams.get(g, 0) + 1
    return grams


def bigram_cosine(a: str, b: str) -> float:
    """正規化→バイグラム頻度ベクトルのコサイン類似度（0.0〜1.0）。"""
    ga, gb = bigrams(normalize(a)), bigrams(normalize(b))
    if not ga or not gb:
        return 0.0
    dot = sum(v * gb.get(k, 0) for k, v in ga.items())
    if dot == 0:
        return 0.0
    na = sum(v * v for v in ga.values()) ** 0.5
    nb = sum(v * v for v in gb.values()) ** 0.5
    return dot / (na * nb)
