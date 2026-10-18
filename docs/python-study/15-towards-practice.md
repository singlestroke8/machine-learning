# 第 15 章 実務への橋渡し — dataclass・logging・テスト・環境構築、そして総合演習

頻度: ★★★★☆ — 題材のスクリプトには出てきませんが、**実務 2 年目のコードには当たり前に出てくる** ものを集めました。
最後に、ここまでの全章の知識を使う総合演習があります。

---

## 15.1 dataclass — 辞書の代わりに「型の付いた入れ物」

### 題材のコードでは

チャンクは辞書で表されています（`scripts/build_kb.py:65-72`）。

```python
chunks.append({
    "page_key": meta["page_key"],
    "title": meta["title"],
    "heading": heading,
    "text": part,
})
```

辞書は手軽ですが、`c["heding"]`（打ち間違い）に気付けない・どんなキーがあるか定義を見ないと分からない、という弱点があります（第 9 章）。

### dataclass で書くと

```python
from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    page_key: str
    title: str
    heading: str
    text: str
    id: str = ""                                   # 既定値のある項目は後ろに
    tags: list[str] = field(default_factory=list)  # リストの既定値はこう書く（第 4 章の罠の対策）


c = Chunk(page_key="price", title="料金", heading="月謝", text="月 8,000 円です。")
print(c.heading)        # 属性で読める。打ち間違えるとエディタが指摘し、実行時も AttributeError
print(c)                # Chunk(page_key='price', title='料金', heading='月謝', text='月 8,000 円です。', id='', tags=[])
print(asdict(c))        # 辞書に戻せる → json.dumps できる
```

`@dataclass` を付けると、型ヒント付きで並べた項目から **`__init__`・`__repr__`・`__eq__` を自動で作って** くれます（第 10 章の特殊メソッド）。

| オプション | 意味 |
|---|---|
| `@dataclass(frozen=True)` | 作った後に属性を変更できない（変更しようとするとエラー）。値オブジェクトに |
| `@dataclass(slots=True)` | メモリを節約し、存在しない属性への代入を防ぐ（3.10+） |
| `field(default_factory=list)` | インスタンスごとに新しいリストを作る |

> **比較**: PHP 8 のコンストラクタプロモーション + `readonly` クラス、TypeScript の `interface` + オブジェクトリテラルに近い役割です。

**使い分けの目安**: JSON や DB と行き来する生データ → 辞書 / TypedDict、プログラムの中で受け渡すデータ → dataclass。

---

## 15.2 logging — print の卒業

題材のスクリプトは `print` で結果を表示しています。**人が実行して結果を見るツール** ならそれで十分です。
一方、サーバや定期実行のバッチでは `logging` を使います。

```python
import logging

logger = logging.getLogger(__name__)      # モジュールごとにロガーを 1 つ。名前はモジュール名


def embed(texts: list[str]) -> None:
    logger.debug("embedding %d texts", len(texts))      # 開発時だけ見たい詳細
    logger.info("cache hits: %d", 120)                   # 通常の進捗
    logger.warning("dimension mismatch: %d", 1024)       # 注意
    logger.error("API failed")                            # エラー


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    embed(["a", "b"])
```

| print と比べた利点 | |
|---|---|
| レベルで出し分けられる | 本番は INFO 以上、調査中は DEBUG まで、を **コードを変えずに** 切り替えられる |
| 時刻・モジュール名が自動で付く | 「いつ・どこで」が分かる |
| 出力先を変えられる | 画面・ファイル・CloudWatch など |
| 標準エラー出力に出る | 標準出力（結果のデータ）とログが混ざらない |

- **ライブラリ側（部品のモジュール）は `logger.xxx()` を呼ぶだけ**、設定（`basicConfig`）は **`main()` 側で 1 回だけ** 行います。
- `logger.info("hits: %d", n)` のように **`%` 形式で引数を分けて渡す** のが流儀です。
  f 文字列 `logger.info(f"hits: {n}")` でも動きますが、表示されないレベルのログでも文字列を作ってしまうため、ログでは `%` 形式が推奨されます。
- 例外を捕まえたときは `logger.exception("API failed")` でトレースバックごと記録できます（第 8 章の「握りつぶさない」）。

---

## 15.3 テスト — pytest

実務 2 年目に求められる力の中で、独学で一番抜けやすいのがテストです。Python では **pytest** が事実上の標準です。

### 最小のテスト

`scripts/build_kb.py` の `split_long()` をテストするとします。

```python
# tests/test_build_kb.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_kb import CHUNK_MIN, split_long  # noqa: E402


def test_short_text_is_one_chunk():
    text = "あ" * 100
    assert split_long(text) == [text]


def test_long_text_is_split_at_paragraphs():
    para = "あ" * 300
    parts = split_long(f"{para}\n{para}")
    assert parts == [para, para]          # 300 + 300 > 450 なので 2 つに分かれる


def test_tiny_fragment_is_dropped():
    assert split_long("あ" * (CHUNK_MIN - 1)) == []
```

```bash
pip install pytest
python -m pytest -q              # tests/ 以下の test_*.py を自動で探して実行
```

- ファイル名は `test_*.py`、関数名は `test_*`。それだけで pytest が見つけてくれます。
- 確認は **ただの `assert`**。失敗すると、pytest が左右の値の違いを詳しく表示してくれます（PHPUnit の `assertEquals` のようなメソッドは不要）。
- PHP の PHPUnit、JS の Jest / Vitest に当たります（PivotAggregationSystem の `src/**/*.test.ts` は Vitest）。

### よく使う機能

```python
import pytest


@pytest.mark.parametrize(                     # 同じテストを入力を変えて何度も
    ("text", "expected"),
    [("ＡＩ　チャット", "aiチャット"), ("三味線（津軽）", "三味線津軽")],
)
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_bad_input_type():
    with pytest.raises(ValueError):           # 例外が出ることを確かめる
        emb.embed(["a"], "documnet")


def test_write_json(tmp_path):                # tmp_path: テストごとの使い捨てフォルダ（第 7 章のファイル処理のテストに）
    path = tmp_path / "out.json"
    ...


def test_score():
    assert cosine([1, 0], [1, 1]) == pytest.approx(0.7071, abs=1e-4)   # 小数の比較は approx
```

### テストしやすいコードとは

- **入出力（ファイル・API・DB）と計算を分ける**: `split_long()` や `bigram_cosine()` は文字列を受け取って値を返すだけなので、テストが簡単です。
  `main()` に全部書いてあるとテストしにくくなります。題材のコードが「部品の関数 + main」に分かれているのはこのためでもあります。
- **外部サービスは差し替えられるようにする**: 第 10 章の `MockEmbedder`、`decide()` が `router` を引数で受け取る設計。
- **`main(argv)` で引数を受け取れるようにする**: 第 13 章。

---

## 15.4 環境構築 — venv と依存の管理

### 仮想環境（venv）

Python は、パッケージを **プロジェクトごとの仮想環境** に入れるのが基本です（PHP の `vendor/`、Node の `node_modules/` に相当するものを自分で作るイメージ）。

```bash
python3 -m venv .venv              # .venv フォルダに仮想環境を作る
source .venv/bin/activate          # 有効化（Windows は .venv\Scripts\activate）
pip install -r requirements.txt    # この環境にだけ入る
deactivate                         # 無効化
```

`.venv/` は `.gitignore` に入れ、コミットしません。

### 依存の書き方

| ファイル | 用途 |
|---|---|
| `requirements.txt` | インストールするパッケージの一覧（このリポジトリの `docker/python/requirements.txt`） |
| `pyproject.toml` | プロジェクトの設定を 1 か所にまとめる新しい標準。依存・ツールの設定（ruff, pytest, mypy）もここに書ける |

最近は **uv** というツール（Rust 製で高速）で、venv の作成と依存の管理をまとめて行うプロジェクトが増えています。

```bash
uv venv                     # 仮想環境を作る
uv pip install pytest       # インストール（pip より大幅に速い）
uv run python script.py     # 仮想環境で実行
```

### このリポジトリのやり方

machine-learning は venv の代わりに **Docker コンテナ** を使っています（`make py ARGS="scripts/xxx.py"`）。
ホストの Python を汚さず、全員が同じバージョン（Python 3.11）・同じパッケージで実行できる、という venv と同じ目的を、より強く実現する方法です。

---

## 15.5 コード品質ツール

| ツール | 役割 | PHP / JS での相当 |
|---|---|---|
| **ruff** | リンター + フォーマッター（`ruff check` / `ruff format`）。import の並べ替えも | PHP_CodeSniffer / PHP-CS-Fixer、ESLint + Prettier |
| **mypy** / **pyright** | 型チェック（第 9 章） | PHPStan / Psalm、`tsc` |
| **pytest** | テスト | PHPUnit、Jest / Vitest |
| **pre-commit** | コミット前に上記を自動実行 | husky + lint-staged |

```bash
ruff check .            # 問題の指摘（未使用の import、未定義の名前、E402 など）
ruff check --fix .      # 自動で直せるものは直す
ruff format .           # 整形（行の長さ・クォート・空行を統一）
```

第 6 章の `# noqa: E402` は ruff（など）への指示でした。このツールを入れて、指摘をゼロにする習慣を付けるだけで、
コードは一気に「実務の Python」らしくなります。

---

## 15.6 ジェネレータ関数 — yield

第 5 章のジェネレータ式を、関数で書く方法です。実務のデータ処理でよく使います。

```python
from collections.abc import Iterator


def iter_batches(items: list[str], size: int) -> Iterator[list[str]]:
    """items を size 件ずつ区切って 1 つずつ返す。"""
    for start in range(0, len(items), size):
        yield items[start : start + size]      # return の代わりに yield


for batch in iter_batches(texts, 128):
    res = client.embed(batch, ...)
```

- `yield` がある関数は、呼ぶと **ジェネレータ** を返し、for で回すたびに次の `yield` まで実行されます。
- embedders.py の `for start in range(0, len(misses), BATCH): idx = misses[start : start + BATCH]` を、
  名前の付いた部品として切り出した形です。
- 巨大なファイルを 1 行ずつ処理する、API のページングを順に読む、といった **「全部をメモリに載せずに順に流す」** 処理に向いています。

（第 12 章の `@contextmanager` の `yield` も、この仕組みを利用しています。）

---

## 15.7 この教材で扱わなかったもの

実務 2 年目までに出会う可能性が高いものを、名前と一言だけ挙げておきます。

| テーマ | 一言 |
|---|---|
| `async` / `await`（asyncio） | JS と同じ考え方の非同期処理。FastAPI や HTTP クライアント（httpx）で使う。JS の Promise を知っていれば理解は早い |
| `requests` / `httpx` | HTTP クライアントの事実上の標準。ref_embedders の `urllib.request` より圧倒的に書きやすい |
| pandas / NumPy | 表データ・数値計算。データ分析・機械学習では必須 |
| `pydantic` | 型ヒントから入力の検証を行うライブラリ。FastAPI の土台 |
| `enum.Enum` | 選択肢の定義（`"document"` / `"query"` のような文字列の代わり） |
| `functools`（`lru_cache`, `partial`） | 関数の結果のキャッシュ、引数の一部固定 |
| `match` 文 | 構造でのパターンマッチ（3.10+） |
| パッケージの公開 | `pyproject.toml` と PyPI |

---

## 15.8 「Python らしさ」のセルフレビュー表

自分のコードを見直すときのチェックリストです。各項目の後ろは、扱った章です。

- [ ] `for i in range(len(xs))` と書いていないか → `enumerate` / `zip`（3 章）
- [ ] 空のリストに append するだけの for 文 → 内包表記（5 章）
- [ ] `if x == True` / `if len(xs) > 0` → `if x` / `if xs`（1 章）
- [ ] `== None` → `is None`（1 章）
- [ ] 関数の既定値にリスト・辞書を置いていないか（4 章）
- [ ] 1 要素のタプルのカンマ（2 章）
- [ ] `open()` を `with` なしで使っていないか / `encoding="utf-8"` を書いたか（7 章）
- [ ] `except Exception: pass` で握りつぶしていないか（8 章）
- [ ] SQL に f 文字列で値を埋め込んでいないか（12 章）
- [ ] 正規表現に外部の値を埋め込むとき `re.escape` したか（11 章）
- [ ] お金の計算に float を使っていないか → `Decimal`（14 章）
- [ ] 関数に型ヒントと docstring があるか（9 章・1 章）
- [ ] `if __name__ == "__main__":` + `main()` になっているか（6 章）
- [ ] ruff と mypy の指摘がゼロか（15 章）

---

## 15.9 総合演習 — Tier0 照合ツールを作る

本番ボットの最初の段（Tier0: 文字バイグラムでの FAQ 照合）の縮小版を、**1 本のコマンドラインツール + テスト** として作ります。
`reference/okubo-shamisen/textmatch.py` と `eval.py` の `best_faq_match_t0()` を参考にして構いません。

### 仕様

**ファイル: `faq_match.py`**

1. `FaqEntry`（`id: str`, `questions: list[str]`, `answer: str`）と `Match`（`faq_id: str`, `score: float`, `matched_question: str`）を dataclass で定義する
2. `normalize(text) -> str` — NFKC、小文字化、空白・記号の除去
3. `bigram_cosine(a, b) -> float` — 正規化した 2 文の文字バイグラムのコサイン類似度。`collections.Counter` を使う
4. `load_faq(path: Path) -> list[FaqEntry]` — 下の形式の JSON を読む
5. `best_match(question, faq, threshold=0.85) -> Match | None` — 最も近い質問文を探し、閾値以上なら返す。閾値が 0〜1 の範囲外なら `ValueError`
6. コマンドライン: `python3 faq_match.py faq.json "質問" [--threshold 0.85] [-v]`
   - 一致したら `fee (0.92): 月 8,000 円です。` と表示して終了コード 0
   - 一致しなければ `NO_MATCH` と表示して終了コード 1
   - `-v` で全候補のスコアを DEBUG ログに出す
   - FAQ ファイルが無ければ argparse 形式のエラー

```json
[
  {"id": "fee", "questions": ["月謝はいくらですか？"], "answer": "月 8,000 円です。"},
  {"id": "trial", "questions": ["体験レッスンはありますか？"], "answer": "あります。"}
]
```

**ファイル: `test_faq_match.py`** — pytest で、少なくとも次を確かめる

- `normalize` の複数パターン（parametrize）
- 同じ文の類似度が 1、無関係な文の類似度が低い
- 言い換えで正しい FAQ が返る / 閾値未満で `None` / 不正な閾値で `ValueError`
- `load_faq` を `tmp_path` で
- `main()` の出力を `capsys` で

### 追加の課題（考える）

作ったツールで、次の 2 つを試してください。

```bash
python3 faq_match.py faq.json "月謝はいくら？" --threshold 0.7
python3 faq_match.py faq.json "オンラインレッスンはありますか" --threshold 0.7
```

1 つ目は正しく `fee` になりますが、2 つ目（**FAQ に載っていない質問**）も `trial`（体験レッスン）に一致してしまいます。
閾値を 0.85 に上げると 2 つ目は弾けますが、今度は 1 つ目の言い換えも弾かれます。

これは CLAUDE.md に書かれている **「閾値 1 本の限界」** そのものです。本番ボットはこの問題を、
上位候補を LLM に渡して判定させる段（Tier1r）を足すことで解決しました。
あなたなら、このツールの範囲でどんな工夫ができるか考えてみてください（解答例の最後にヒントがあります）。

---

## 総合演習の解答例

以下は、ruff・mypy の指摘ゼロ、pytest 11 件がすべて通ることを確認した解答例です。

<details>
<summary>faq_match.py</summary>

```python
"""FAQ の質問文と、利用者の質問を文字バイグラムで照合する（本番ボットの Tier0 の縮小版）。

使い方:
    python3 faq_match.py faq.json "月謝はいくらですか"
    python3 faq_match.py faq.json "月謝はいくら？" --threshold 0.7 -v

faq.json の形式:
    [{"id": "fee", "questions": ["月謝はいくらですか？"], "answer": "月 8,000 円です。"}, ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 空白と、日本語・英語の主な記号。照合の前に取り除く
_STRIP = re.compile(r"[\s。、．，・！!？?「」『』()（）\[\]〜~\-/／…:：;；\"']+")


@dataclass(frozen=True)
class FaqEntry:
    """FAQ の 1 エントリ。questions は同じ答えになる言い換えの一覧。"""

    id: str
    questions: list[str]
    answer: str


@dataclass(frozen=True)
class Match:
    """照合の結果。どのエントリの、どの質問文に、どれだけ近かったか。"""

    faq_id: str
    score: float
    matched_question: str


def normalize(text: str) -> str:
    """NFKC で全角・半角を揃え、小文字にし、空白と記号を除く。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return _STRIP.sub("", text)


def bigrams(text: str) -> Counter[str]:
    """文字バイグラムの出現回数。1 文字なら、その文字自身を 1 回。"""
    if len(text) < 2:
        return Counter([text]) if text else Counter()
    return Counter(text[i : i + 2] for i in range(len(text) - 1))


def bigram_cosine(a: str, b: str) -> float:
    """2 つの文の、正規化後のバイグラム頻度ベクトルのコサイン類似度（0.0〜1.0）。"""
    ga, gb = bigrams(normalize(a)), bigrams(normalize(b))
    if not ga or not gb:
        return 0.0
    dot = sum(v * gb[k] for k, v in ga.items())  # Counter は無いキーを 0 として返す
    na = sum(v * v for v in ga.values()) ** 0.5
    nb = sum(v * v for v in gb.values()) ** 0.5
    return dot / (na * nb)


def load_faq(path: Path) -> list[FaqEntry]:
    """JSON を読み、FaqEntry のリストにする。必須のキーが無ければ KeyError で止める。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        FaqEntry(id=e["id"], questions=e["questions"], answer=e["answer"]) for e in raw
    ]


def best_match(
    question: str, faq: list[FaqEntry], threshold: float = 0.85
) -> Match | None:
    """最も近い質問文を探し、類似度が threshold 以上ならその結果を、無ければ None を返す。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold は 0〜1: {threshold}")

    best: Match | None = None
    for entry in faq:
        for q in entry.questions:
            score = bigram_cosine(question, q)
            logger.debug("%s %.3f %s", entry.id, score, q)
            if best is None or score > best.score:
                best = Match(entry.id, score, q)

    if best is None or best.score < threshold:
        return None
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FAQ と質問を文字バイグラムで照合する")
    ap.add_argument("faq", type=Path, help="FAQ の JSON ファイル")
    ap.add_argument("question", help="照合する質問")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="これ以上なら一致とみなす(既定 0.85)",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="全候補のスコアを表示する"
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    try:
        faq = load_faq(args.faq)
    except FileNotFoundError:
        ap.error(f"ファイルがありません: {args.faq}")

    match = best_match(args.question, faq, args.threshold)
    if match is None:
        print("NO_MATCH")
        return 1

    answer = next(e.answer for e in faq if e.id == match.faq_id)
    print(f"{match.faq_id} ({match.score:.2f}): {answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

</details>

<details>
<summary>test_faq_match.py</summary>

```python
"""faq_match.py のテスト。実行: python3 -m pytest -q"""

import json

import pytest

from faq_match import FaqEntry, best_match, bigram_cosine, load_faq, main, normalize

FAQ = [
    FaqEntry(
        "fee", ["月謝はいくらですか？", "料金を教えてください"], "月 8,000 円です。"
    ),
    FaqEntry("trial", ["体験レッスンはありますか？"], "あります。"),
]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("月謝は、いくら？", "月謝はいくら"),
        ("ＡＩ　チャット", "aiチャット"),
        ("三味線（津軽）", "三味線津軽"),
        ("", ""),
    ],
)
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_bigram_cosine_same_text_is_one():
    assert bigram_cosine("月謝はいくら", "月謝はいくら？") == pytest.approx(1.0)


def test_bigram_cosine_unrelated_is_low():
    assert bigram_cosine("月謝はいくら", "駐車場の場所") < 0.3


def test_best_match_finds_paraphrase():
    m = best_match("月謝はいくらですか", FAQ)
    assert m is not None
    assert m.faq_id == "fee"
    assert m.matched_question == "月謝はいくらですか？"


def test_best_match_returns_none_below_threshold():
    assert best_match("オンラインレッスンはありますか", FAQ) is None


def test_best_match_rejects_bad_threshold():
    with pytest.raises(ValueError):
        best_match("月謝", FAQ, threshold=1.5)


def test_load_faq(tmp_path):
    path = tmp_path / "faq.json"
    path.write_text(
        json.dumps(
            [{"id": "fee", "questions": ["月謝は？"], "answer": "8,000 円"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert load_faq(path) == [FaqEntry("fee", ["月謝は？"], "8,000 円")]


def test_main_prints_answer(tmp_path, capsys):
    path = tmp_path / "faq.json"
    path.write_text(
        json.dumps(
            [{"id": "fee", "questions": ["月謝は？"], "answer": "8,000 円"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert main([str(path), "月謝は"]) == 0
    assert capsys.readouterr().out == "fee (1.00): 8,000 円\n"
```

</details>

<details>
<summary>解答例のポイント</summary>

| 箇所 | 使った知識 | 章 |
|---|---|---|
| ファイル冒頭の docstring に使い方と入力形式 | docstring | 1, 6 |
| `@dataclass(frozen=True)` | dataclass（作った後に変えない値として） | 15 |
| `_STRIP = re.compile(...)` | 正規表現の事前準備・内部用の `_` | 11, 6 |
| `Counter(text[i : i + 2] for i in ...)` | ジェネレータ式・Counter | 5, 14 |
| `gb[k]`（`get` 不要） | Counter は無いキーを 0 で返す | 14 |
| `best: Match \| None = None` | 型ヒントの `\| None` | 9 |
| `raise ValueError(...)` | 呼び出し側のミスを早く知らせる | 8 |
| `logger.debug("%s %.3f %s", ...)` | logging の `%` 形式 | 15 |
| `main(argv) -> int` + `sys.exit(main())` | テストしやすい main・終了コード | 13, 8 |
| `ap.error(...)` | 利用者のミスは argparse 形式で | 13 |
| `next(e.answer for e in faq if e.id == ...)` | ジェネレータ式から最初の 1 件 | 5, 14 |
| `tmp_path` / `capsys` / `parametrize` / `approx` | pytest の道具 | 15 |

</details>

<details>
<summary>追加の課題のヒント</summary>

- **2 つの閾値を使う**: 0.85 以上は即答、0.6〜0.85 は「もしかして: 体験レッスンのことですか？」と **確認を返す**。
  間違えて答えるより、聞き返す方が害が小さい
- **否定的な手がかりを見る**: FAQ に無い重要語（「オンライン」）が質問に含まれていたら、スコアを下げる
- **評価セットを作る**: `corpus/eval/positive.md`（答えるべき言い換え）と `negative.md`（答えてはいけない質問）のような
  データを用意し、閾値を変えながら「正答カバー率」と「誤回答数」を測る — eval.py の `sweep()` / `recommend()` がまさにこれです

最後の方法が、本番ボットの開発で実際に取られたやり方です。**勘ではなく、評価データで決める** — これが、この教材の題材コード全体を貫く考え方です。

</details>

---

## おわりに

ここまでで、このリポジトリと PivotAggregationSystem の Python スクリプトは、ほぼすべての行が読めるようになっているはずです。
次のステップとしておすすめするのは:

1. **題材のファイルを、何も見ずに頭から読む**。分からない行があれば、その章に戻る
2. **題材に小さな改善を入れてみる**: `zip(..., strict=True)` を足す、`input_type` を `Literal` にする、`split_long` のテストを書く
3. **自分の業務の小さな作業を Python のツールにする**: CSV の集計、ログの整形、API の呼び出し。argparse + pathlib + csv / json の組み合わせで大半が書けます
4. その次に、FastAPI などのフレームワークへ

実務 2 年目のコードは、特別な技を使っているわけではありません。
**この教材で扱った基本を、適切な場所で、迷わず選べること** — それが「書ける」ということです。
