# 第 9 章 型ヒント

頻度: ★★★★☆ — 題材のすべての関数に付いています。今の実務の Python では「書くのが当たり前」です。

この章で読めるようになるコード:

```python
def parse_page(path: Path) -> tuple[dict[str, str], str]:
out: list[list[float] | None] = [None] * len(texts)
def decide(..., router: "Router | None") -> tuple[str, str | None, list[str]]:
return out  # type: ignore[return-value]
from __future__ import annotations
```

---

## 9.1 型ヒントの書き方

### 題材コード

`scripts/build_kb.py:22, 32, 52, 88`

```python
def parse_page(path: Path) -> tuple[dict[str, str], str]:
def split_long(text: str) -> list[str]:
def chunk_page(meta: dict[str, str], body: str) -> list[dict]:
def main() -> None:
```

`scripts/build_kb.py:37, 54`

```python
parts: list[str] = []
chunks: list[dict] = []
```

### 解説

| 場所 | 書き方 | 意味 |
|---|---|---|
| 引数 | `text: str` | 引数 `text` は文字列 |
| 戻り値 | `-> list[str]` | 文字列のリストを返す |
| 戻り値なし | `-> None` | 何も返さない（`main()` など） |
| 変数 | `parts: list[str] = []` | この変数には文字列のリストを入れる |

変数の型ヒントは、**空のリストや辞書で初期化するとき** に特によく書きます。
`parts = []` だけでは「何のリストか」が分からないからです。

`tuple[dict[str, str], str]` は **入れ子** で、「(文字列→文字列の辞書, 文字列) の 2 要素のタプル」と読みます。
`parse_page()` が `(meta, body)` を返すことが、関数の中を読まなくても分かります。

---

## 9.2 型ヒントは「実行時にはチェックされない」

**これが TypeScript・PHP と比べたときの最大の違いです。**

```python
def double(x: int) -> int:
    return x * 2

>>> double("ab")
'abab'            # エラーにならない！
```

Python の型ヒントは **注釈（メモ）** で、実行時には何もチェックしません。

| | 型を書く | 誰がチェックする | 違反すると |
|---|---|---|---|
| **TypeScript** | `x: number` | コンパイラ（`tsc`） | コンパイルエラー。実行時は型が消える |
| **PHP**（`declare(strict_types=1)`） | `int $x` | 実行時のエンジン | 実行時に `TypeError` |
| **Python** | `x: int` | **別のツール**（mypy / pyright）。エディタが裏で実行 | ツールが警告。**実行は普通に進む** |

TypeScript と同じく「書いた型を **静的解析ツール** が検査する」仕組みです。
違いは、TS ではコンパイラが必須なのに対し、Python ではチェックを **実行しなくてもプログラムは動く** 点です。

それでも書く理由:

1. **読む人のため** — 関数の中を読まなくても、何を渡して何が返るか分かる
2. **エディタのため** — VS Code（Pylance）が補完してくれる・間違いに波線を出してくれる
3. **検査ツールのため** — `mypy` や `pyright` を CI で回すと、型の不一致をバグになる前に見つけられる

---

## 9.3 よく使う型の書き方

### 基本

| 型ヒント | 意味 | 題材 |
|---|---|---|
| `int` `float` `str` `bool` | 基本の型 | 多数 |
| `list[str]` | 文字列のリスト | build_kb |
| `dict[str, int]` | キーが文字列・値が整数の辞書 | textmatch の `bigrams()` |
| `tuple[str, float]` | 2 要素のタプル（それぞれの型） | eval の `best_faq_match_t0()` |
| `tuple[int, ...]` | 任意の長さの整数のタプル | |
| `set[str]` | 文字列のセット | |
| `Path` | クラスもそのまま型として使える | build_kb |
| `random.Random` | 同上 | dummy_csv |

`int` が期待される場所に `float` を渡すと怒られますが、`float` が期待される場所に `int` を渡すのは OK、という
特別ルールもあります（`cosine(a: list[float], ...)` に `[1, 2]` を渡しても警告されない）。

### None かもしれない — `X | None`

`reference/okubo-shamisen/eval.py:191`

```python
def recommend(rows: list[dict]) -> float | None:
    """負例の誤回答ゼロでカバー率最大のτ（同率なら低い方＝カバー広い方）。"""
    ok = [r for r in rows if r["false_answers"] == 0]
    if not ok:
        return None
    ...
```

`float | None` は「float か None」。条件を満たす τ が無いときに `None` を返すからです。
TypeScript の `number | null`、PHP の `?float` と同じ意味です。

`X | None` の書き方は **Python 3.10 から**。古いコードでは同じ意味で `Optional[float]`（`from typing import Optional`）と書かれています。

**呼び出し側は None を考慮する必要がある** ことが、型ヒントから分かります。実際 eval.py の呼び出し側は、

```python
f"- **推奨 τ0 = {tau0}**（...）" if tau0 is not None else "- 誤回答ゼロにできるτがない（要調査）",
```

と `None` の場合を分けて表示しています。型チェッカーは、この分岐を書き忘れて `tau0 + 0.05` のように使うと警告してくれます。

### 入れ子の `| None`

`scripts/embedders.py:50`

```python
out: list[list[float] | None] = [None] * len(texts)
```

「要素が `list[float]`（ベクトル）か `None` であるリスト」。最初は全部 `None` の席を作り、後で埋めていくからです（第 2 章）。

---

## 9.4 `# type: ignore` — 型チェッカーを黙らせる

### 題材コード

`scripts/embedders.py:44, 80`

```python
def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
    ...
    return out  # type: ignore[return-value]
```

### 解説

- 関数の宣言では「`list[list[float]]`（None を含まない）を返す」と約束しています。
- しかし `out` の型は `list[list[float] | None]`（None を含みうる）です。
- 処理の流れを追えば、最後には全部の席が埋まっていて None は残らないのですが、
  **型チェッカーにはそこまで分からない** ので「戻り値の型が違う」と警告します。
- `# type: ignore[return-value]` は「この行の `return-value` に関する警告は分かったうえで無視する」という指示です。

角カッコで **無視する警告の種類を限定** しているのが良い書き方です。`# type: ignore` だけだと、
その行の別の問題まで隠してしまいます。

> 本当は、最後に `assert all(v is not None for v in out)` を入れたり、
> `return [v for v in out if v is not None]` のように型が絞り込まれる書き方にしたりすれば
> ignore 自体を消せます。ignore は「直すコストに見合わないとき」の最終手段です。

---

## 9.5 クォートで囲んだ型ヒントと `from __future__ import annotations`

### 題材コード

`reference/okubo-shamisen/eval.py:123-124`

```python
def decide(question: str, qvec: list[float], faq: list[dict], cfg: dict,
           router: "Router | None") -> tuple[str, str | None, list[str]]:
```

`PivotAggregationSystem/scripts/generate_dummy_csv.py:36`

```python
from __future__ import annotations
```

### 解説

型ヒントは、通常は **関数を定義した時点で評価** されます。そのため、

- まだ定義されていないクラス（後ろの行で定義されるクラスなど）を型ヒントに書くと `NameError` になる
- 古い Python で `date | None` と書くと、`|` が使えずにエラーになる

という問題がありました。回避策が 2 つあります。

1. **型ヒントを文字列にする**: `"Router | None"` — 文字列なので評価されず、型チェッカーだけが中身を読みます（前方参照）。
   eval.py の `Router` クラスは `decide()` より上で定義されているので、実は 3.10 以降なら文字列にしなくても動きます。
2. **`from __future__ import annotations`** をファイル先頭に書く — そのファイルの **すべての型ヒントが評価されなくなる**
   （自動的に文字列扱い）。dummy_csv はこれを書いているので、`date | None` を Python 3.7〜3.9 でも書けます。

`from __future__ import annotations` は、型ヒントを多用するコードの先頭によく書かれています。「おまじない」として見かけたら、
「型ヒントを評価しないようにする指定だな」と思い出してください。

---

## 9.6 `list[dict]` の限界と TypedDict

### 題材コード

`scripts/build_kb.py:52`

```python
def chunk_page(meta: dict[str, str], body: str) -> list[dict]:
```

### 解説

`list[dict]` は「辞書のリスト」としか言っていません。どんなキーがあるのか、値の型は何かが分かりません。
`c["heding"]`（打ち間違い）と書いても、型チェッカーは気付けません。

**辞書のキーと値の型まで表す** には `TypedDict` を使います。

```python
from typing import TypedDict


class Chunk(TypedDict):
    id: str
    page_key: str
    title: str
    heading: str
    text: str


def chunk_page(meta: dict[str, str], body: str) -> list[Chunk]:
    ...
```

- 実行時はただの `dict` のまま（JSON にそのまま書ける）
- 型チェッカーは `c["heding"]` を「そんなキーは無い」と指摘してくれる
- TypeScript の `interface Chunk { id: string; ... }` に一番近い書き方です

さらに、辞書ではなく **属性でアクセスする** オブジェクトにしたいなら `@dataclass` を使います（第 15 章）。
「JSON と行き来する生データは TypedDict、プログラム内で扱うデータは dataclass」という使い分けが一般的です。

---

## 9.7 typing モジュールのよく使う道具

| 書き方 | 意味 | 使いどころ |
|---|---|---|
| `Any` | 何でもよい（チェックしない） | 型が付けられない外部データ。多用は禁物 |
| `Literal["document", "query"]` | この値のどれか | 決まった文字列しか受け付けない引数 |
| `Callable[[str, list[dict]], str \| None]` | この形の関数 | 関数を引数で受け取るとき |
| `Iterable[str]` / `Sequence[str]` | for で回せるもの / 添字で読めるもの | 引数を `list` に限定したくないとき |
| `TypedDict` | キーが決まった辞書 | 9.6 |
| `NamedTuple` | 名前付きのタプル | 第 4 章 問 6 |

### Literal で input_type を守る

embedders.py は、`input_type` が `"document"` か `"query"` かを **実行時に** チェックしています。

```python
def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
    if input_type not in ("document", "query"):
        raise ValueError(f"input_type は document か query: {input_type}")
```

`Literal` を使えば、**書いた時点で** エディタが間違いを指摘してくれます。

```python
from typing import Literal

InputType = Literal["document", "query"]   # 型に名前を付けておける（型エイリアス）

def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
    ...

emb.embed(texts, "documnet")   # ← エディタが赤い波線を出す
```

型ヒントは実行時には効かないので、**実行時チェック（`raise ValueError`）は残しておく** のが堅実です。
外部から何が来るか分からない入口では両方、内部の関数どうしなら型ヒントだけ、というのが一般的な落としどころです。

### 引数は広く、戻り値は具体的に

```python
from collections.abc import Iterable

def total(xs: Iterable[float]) -> float:    # リストでもジェネレータでもタプルでも受け取れる
    return sum(xs)
```

引数を `list[float]` にすると、ジェネレータ式やタプルを渡したときに型チェッカーが警告します。
**受け取る側は「最低限必要な性質」で、返す側は「具体的な型」で** 書くと使いやすい関数になります。

---

## 9.8 型チェッカーを動かしてみる

```bash
pip install mypy             # または pyright
mypy scripts/                # フォルダごと検査
```

```python
# check.py
def split_long(text: str) -> list[str]:
    return text.split("\n")

n: int = split_long("a\nb")   # list[str] を int の変数に入れている
```

```
$ mypy check.py
check.py:4: error: Incompatible types in assignment (expression has type "list[str]", variable has type "int")
```

VS Code に Python 拡張（Pylance）を入れていれば、ツールを動かさなくてもエディタ上で同じ指摘が出ます。
**まずはエディタの波線を消す習慣** から始めるのがおすすめです。

---

## 章末問題

### 問 1（読解）

次の型ヒントを日本語で説明してください。

1. `def bigrams(text: str) -> dict[str, int]:`
2. `def rank_faq_t1(qvec: list[float], faq: list[dict]) -> list[tuple[str, float]]:`
3. `def ymd(d: date | None) -> str:`
4. `def sweep(results_pos: list[tuple[bool, float]], results_neg: list[float], taus: list[float]) -> list[dict]:`

### 問 2（書く）

次の関数に型ヒントを付けてください。

```python
def chunk_to_text(chunk):
    if chunk["heading"]:
        return f"{chunk['heading']}\n{chunk['text']}"
    return chunk["text"]

def pick(rng, choices, weights):
    return rng.choices(choices, weights=weights)[0]
```

### 問 3（考える）

次のコードは実行するとどうなりますか？ 型チェッカー（mypy）にかけるとどうなりますか？

```python
def ratio(hit: int, total: int) -> float:
    return hit / total

print(ratio("3", 4))
```

### 問 4（書く）

`best_faq_match_t0()` は `(FAQのid, スコア)` を返します。見つからないときは `("", 0.0)` を返しています。
これを「見つからないときは `None` を返す」に変えるとしたら、戻り値の型ヒントはどう書きますか？
呼び出し側はどう変わりますか？

### 問 5（書く）

dummy_csv の `make_lines()` が返す明細の辞書のうち、`品名`（str）・`数量`（int）・`商品フラグ`（int）だけを持つ
TypedDict `Line` を定義してください（ヒント: キーが日本語や記号を含む場合は、関数形式 `TypedDict("Line", {...})` で書きます）。

---

## 回答

<details>
<summary>問 1 の回答</summary>

1. 文字列を受け取り、「文字列 → 整数」の辞書を返す（バイグラム → 出現回数）
2. 浮動小数点数のリスト（ベクトル）と辞書のリスト（FAQ）を受け取り、「(文字列, 小数) のタプル」のリストを返す（(id, スコア) の一覧）
3. 日付か None を受け取り、文字列を返す（None なら空文字）
4. 「(真偽値, 小数)」のリスト・小数のリスト・小数のリストを受け取り、辞書のリストを返す

</details>

<details>
<summary>問 2 の回答</summary>

```python
import random


def chunk_to_text(chunk: dict) -> str:
    ...


def pick(rng: random.Random, choices: list[str], weights: list[int]) -> str:
    ...
```

`chunk` は 9.6 の `Chunk`（TypedDict）があれば `chunk: Chunk` と書くのがより良いです。
`pick` を文字列以外にも使いたいなら、ジェネリクス（`def pick[T](rng, choices: list[T], ...) -> T:`、Python 3.12+）
という書き方もありますが、まずは具体的な型で十分です。

</details>

<details>
<summary>問 3 の回答</summary>

- **実行すると** `TypeError: unsupported operand type(s) for /: 'str' and 'int'` で止まります。
  型ヒントが原因ではなく、`"3" / 4` という計算ができないからです（型ヒントは実行時に何もしない）。
- **mypy にかけると** 実行する前に `Argument 1 to "ratio" has incompatible type "str"; expected "int"` と指摘されます。

もし関数の中身が `return hit * 2` のように文字列でも動く処理だったら、実行時にはエラーにすらならず、
おかしな値が流れていきます。型チェッカーの価値はそこにあります。

</details>

<details>
<summary>問 4 の回答</summary>

```python
def best_faq_match_t0(query: str, faq: list[dict]) -> tuple[str, float] | None:
```

呼び出し側は `None` を考慮する必要があります。

```python
match = best_faq_match_t0(question, faq)
if match is not None and match[1] >= cfg["tau0"]:
    got0, s0 = match
    return "t0", got0, []
```

`("", 0.0)` のような「空の値」を返す方式は呼び出し側が簡単になる一方、`None` を返す方式は
「見つからなかった」ことを **型で強制的に意識させられる** という利点があります。どちらにも一長一短があります。

</details>

<details>
<summary>問 5 の回答</summary>

```python
from typing import TypedDict

Line = TypedDict("Line", {"品名": str, "数量": int, "商品フラグ": int})

line: Line = {"品名": "送料", "数量": 1, "商品フラグ": 0}
```

`class Line(TypedDict): 品名: str` のようにクラス形式でも日本語の識別子は使えますが、
`"仕入単価(税別)"` のように **カッコや記号を含むキー** はクラス形式では書けないので、関数形式を使います。

</details>
