"""corpus/pages/*.md をチャンク化して .cache/chunks.json に書き出す。

使い方: make py ARGS="scripts/build_kb.py"

API も DB も使わないので、何度実行しても無料。
埋め込みは embedders.py、DB 投入は load_kb.py が担当する。
チャンク化の規則は本番ボット（reference/okubo-shamisen/build_kb.py）と同じ。
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / "corpus/pages"
OUT_PATH = ROOT / ".cache/chunks.json"

CHUNK_MAX = 450  # 見出しを除く本文の最大文字数。長いと意味が薄まり、検索に引っかかりにくくなる
CHUNK_MIN = 60   # これ未満はボタン文言などのノイズとして捨てる


def parse_page(path: Path) -> tuple[dict[str, str], str]:
    """ページ先頭の <!-- key: value --> をメタ情報として取り出し、本文からは取り除く。"""
    text = path.read_text(encoding="utf-8")
    # 例: <!-- page_key: column-equipment --> → {"page_key": "column-equipment"}
    meta = dict(re.findall(r"<!--\s*([a-z_]+):\s*(.*?)\s*-->", text))
    # re.S で改行をまたぐコメントも消す
    body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return meta, body


def split_long(text: str) -> list[str]:
    """CHUNK_MAX を超える本文を、段落（改行）の境目で分ける。

    段落の途中では切らないので、1 段落だけで CHUNK_MAX を超える場合はそのまま 1 チャンクになる。
    """
    parts: list[str] = []
    cur = ""
    for para in text.split("\n"):
        # 次の段落を足すと上限を超えるなら、ここまでを 1 つ確定して新しく始める
        if cur and len(cur) + len(para) > CHUNK_MAX:
            parts.append(cur)
            cur = para
        else:
            cur = f"{cur}\n{para}" if cur else para
    if cur:
        parts.append(cur)
    # 分けた結果、短すぎる切れ端ができたら捨てる
    return [p.strip() for p in parts if len(p.strip()) >= CHUNK_MIN]


def chunk_page(meta: dict[str, str], body: str) -> list[dict]:
    """1 ページ分の本文を、見出しごとのチャンクのリストにする。"""
    chunks: list[dict] = []
    heading = ""  # 今いる見出し。最初の見出しより前（ページ冒頭）は空
    buf: list[str] = []  # 今の見出しの下に溜まっている行

    def flush() -> None:
        """溜まった行を 1 つ以上のチャンクとして確定し、buf を空にする。"""
        text = "\n".join(buf).strip()
        buf.clear()
        if len(text) < CHUNK_MIN:
            return
        for part in split_long(text):
            chunks.append({
                # meta["..."] は、コメント行が欠けたページがあれば KeyError で止まる。
                # 空文字のまま DB の NOT NULL 列に入るのを防ぐため
                "page_key": meta["page_key"],
                "title": meta["title"],
                "heading": heading,
                "text": part,
            })

    # 1. 本文を 1 行ずつ読む
    for line in body.splitlines():
        # 2. 見出し行（"# " 〜 "###### "）が来たら、それまでの塊を確定して見出しを切り替える
        if re.match(r"^#{1,6} ", line):
            flush()
            heading = line.lstrip("#").strip()
        # 3. それ以外の行は今の見出しの下に溜める
        else:
            buf.append(line)
    # 4. 最後の見出しの塊を確定する
    flush()
    return chunks


def main() -> None:
    # 1. 全ページを読み、チャンクを 1 つのリストに集める（ファイル名順で毎回同じ並びになる）
    chunks: list[dict] = []
    for path in sorted(PAGES_DIR.glob("*.md")):
        meta, body = parse_page(path)
        chunks.extend(chunk_page(meta, body))

    # 2. 通し番号を振る（c0000, c0001, ...）。並びが変わらない限り同じ id になる
    for i, c in enumerate(chunks):
        c["id"] = f"c{i:04d}"

    # 3. JSON に書き出す。ensure_ascii=False で日本語をそのまま読める形にする
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")

    # 4. 件数と長さの分布を表示する。規則を変えたときの見比べに使う
    lens = [len(c["text"]) for c in chunks]
    print(f"pages: {len(list(PAGES_DIR.glob('*.md')))} / chunks: {len(chunks)}")
    print(f"text length: min={min(lens)} max={max(lens)} avg={sum(lens) / len(lens):.0f}")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()