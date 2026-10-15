#!/usr/bin/env python3
"""知識ベースのビルド: コーパス → qa-light.json / kb.json

使い方:
    python3 rag/build_kb.py --provider voyage        # 本番（要 VOYAGE_API_KEY）
    python3 rag/build_kb.py --provider mock          # 開発・動作確認用

出力（rag/dist/ 以下、git管理外）:
    qa-light.json  … Tier0用。ウィジェットが読む軽量ファイル（ベクトルなし）
    kb.json        … Tier1/2用。PHPが読む埋め込み入り知識ベース

デプロイ時は2ファイルを child theme 直下へアップロードする。
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedders import make_embedder
from validate_corpus import load_corpus

RAG_DIR = Path(__file__).resolve().parent
PAGES_DIR = RAG_DIR / "corpus/pages"
DIST_DIR = RAG_DIR / "dist"

CHUNK_MAX = 450   # チャンクの最大文字数（見出し除く）
CHUNK_MIN = 60    # これ未満の断片は捨てる（ボタン文言などのノイズ対策）


def parse_page(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    meta = dict(re.findall(r"<!--\s*([a-z_]+):\s*(.*?)\s*-->", text))
    body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return meta, body


def chunk_page(meta: dict[str, str], body: str) -> list[dict]:
    """見出し単位でまとめ、CHUNK_MAX を超えたら分割する。"""
    chunks: list[dict] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        buf.clear()
        if len(text) < CHUNK_MIN:
            return
        # 長すぎる場合は段落境界で分割
        parts: list[str] = []
        cur = ""
        for para in text.split("\n"):
            if cur and len(cur) + len(para) > CHUNK_MAX:
                parts.append(cur)
                cur = para
            else:
                cur = f"{cur}\n{para}" if cur else para
        if cur:
            parts.append(cur)
        for part in parts:
            if len(part.strip()) < CHUNK_MIN:
                continue
            chunks.append({
                "page_key": meta.get("page_key", ""),
                "title": meta.get("title", ""),
                "heading": heading,
                "text": part.strip(),
            })

    for line in body.splitlines():
        if re.match(r"^#{1,6} ", line):
            flush()
            heading = line.lstrip("#").strip()
        else:
            buf.append(line)
    flush()
    return chunks


def round_vec(vec: list[float], nd: int = 4) -> list[float]:
    return [round(x, nd) for x in vec]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["voyage", "mock"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dim", type=int, default=256)
    args = ap.parse_args()

    entries, errors = load_corpus()
    if errors:
        for e in errors:
            print(f"corpus error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- ページ本文のチャンク化 ---
    chunks: list[dict] = []
    for path in sorted(PAGES_DIR.glob("*.md")):
        meta, body = parse_page(path)
        chunks.extend(chunk_page(meta, body))
    # 想定問答の回答もTier2の根拠チャンクに含める。
    # 質問文は先頭1件ではなく全件入れる。1エントリが複数の話題を扱う場合
    # （例: 子供の月謝・付き添い・楽器・通学）、先頭の質問だけだとチャンクの
    # ベクトルがその話題に偏り、他の話題で検索が届かなくなるため。
    for e in entries:
        qs = "\n".join(f"Q. {q}" for q in e.questions)
        chunks.append({
            "page_key": e.link or "faq",
            "title": "よくある質問",
            "heading": e.title,
            "text": f"{qs}\nA. {e.answer}",
        })
    for i, c in enumerate(chunks):
        c["id"] = f"c{i:04d}"

    emb = make_embedder(args.provider, args.model, args.dim)

    # --- 埋め込み ---
    q_texts = [q for e in entries for q in e.questions]
    # Tier1 は「訪問者の質問」と「想定問答の質問文」という質問同士の比較なので、
    # 実行時（rest.php）と同じ query 側で揃える。document にすると同一文でも類似度が下がる。
    q_vecs = emb.embed(q_texts, "query")
    it = iter(q_vecs)
    faq_out = []
    for e in entries:
        vecs = [round_vec(next(it)) for _ in e.questions]
        faq_out.append({
            "id": e.id,
            "answer": e.answer,
            "link": e.link,
            "questions": e.questions,
            "qvecs": vecs,
        })

    c_vecs = emb.embed([f"{c['heading']}\n{c['text']}" if c["heading"] else c["text"] for c in chunks], "document")
    chunks_out = [
        {**c, "vec": round_vec(v)} for c, v in zip(chunks, c_vecs)
    ]

    DIST_DIR.mkdir(exist_ok=True)
    kb = {
        "version": 2,
        "qvec_input_type": "query",
        "provider": emb.provider,
        "model": emb.model,
        "dim": emb.dim,
        "faq": faq_out,
        "chunks": chunks_out,
    }
    kb_path = DIST_DIR / "kb.json"
    kb_path.write_text(json.dumps(kb, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    qa_light = [
        {"id": e.id, "qs": e.questions, "a": e.answer, "link": e.link}
        for e in entries
    ]
    ql_path = DIST_DIR / "qa-light.json"
    ql_path.write_text(json.dumps(qa_light, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    print(f"faq: {len(faq_out)}エントリ / 質問ベクトル {len(q_texts)}件")
    print(f"chunks: {len(chunks_out)}件")
    print(f"{kb_path}: {kb_path.stat().st_size / 1024:.0f} KB")
    print(f"{ql_path}: {ql_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
