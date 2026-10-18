# 第 4 章 関数（引数・戻り値・アンパック・クロージャ・lambda）

頻度: ★★★★★

この章で読めるようになるコード:

```python
def decide(question: str, qvec: list[float], faq: list[dict], cfg: dict,
           router: "Router | None") -> tuple[str, str | None, list[str]]:
meta, body = parse_page(path)
self.client = voyageai.Client(max_retries=3, timeout=60)
return sorted(scored, key=lambda x: -x[1])
```

型ヒント（`: str` や `-> list[str]`）の詳しい話は第 9 章でします。この章では
「**引数名の後ろの `: 型` と、`->` の後ろの戻り値の型は、説明のための注釈**」とだけ覚えておいてください。

---

## 4.1 関数の定義と呼び出し

### 題材コード

`reference/okubo-shamisen/eval.py:146-148`

```python
def light_variants(q: str) -> list[str]:
    """想定問答の質問文と、訪問者が付けがちな軽い前置き・後置き。"""
    return [q, f"あの、{q}", f"{q}教えてください"]
```

### 解説

- `def 関数名(引数):` で定義し、`return` で値を返します。
- `return` が無い関数（または `return` だけ）は **`None` を返します**（JS の `undefined` に当たる）。
- 関数定義の直後の `"""..."""` は docstring（第 1 章）。
- 関数は使う場所より **前に定義されている必要はありません**。呼ばれた時点で存在していれば OK です。
  そのため Python のスクリプトは「部品の関数を上に、`main()` を下に」並べるのが一般的です。

```python
>>> light_variants("月謝はいくら？")
['月謝はいくら？', 'あの、月謝はいくら？', '月謝はいくら？教えてください']
```

評価で「ユーザーが少し言い回しを変えても同じ答えが返るか」を確かめるための、質問の変形を作る関数です。

---

## 4.2 既定値つき引数とキーワード引数

### 題材コード

`scripts/embedders.py:27-33`

```python
class Embedder:
    def __init__(self, model: str = DEFAULT_MODEL, dim: int = DIM):
        ...
        self.client = voyageai.Client(max_retries=3, timeout=60)
```

`scripts/embedders.py:63-68`

```python
res = self.client.embed(
    [texts[i] for i in idx],
    model=self.model,
    input_type=input_type,
    output_dimension=self.dim,
)
```

### 解説

**既定値（デフォルト引数）**: `model: str = DEFAULT_MODEL` は「渡されなければ `DEFAULT_MODEL` を使う」。
`Embedder()` とも `Embedder("voyage-3.5")` とも呼べます。

**キーワード引数**: 呼び出し側で `名前=値` と書くと、**順番に関係なく** 名前で引数を指定できます。

```python
voyageai.Client(max_retries=3, timeout=60)
voyageai.Client(timeout=60, max_retries=3)   # 同じ意味
```

- 位置で渡す引数（`[texts[i] for i in idx]`）を先に、キーワード引数を後に書きます。
- 引数が 3 つを超えたら、**キーワード引数で書く方が読みやすく安全** です。
  `embed(texts, "voyage-3.5", "query", 256)` より `embed(texts, model=..., input_type=..., output_dimension=...)` の方が、
  何をどこに渡しているか一目で分かります。
- `json.dumps(chunks, ensure_ascii=False, indent=1)` のように、**オプション的な引数はキーワードで渡す** のが Python の流儀です。

> **JS / PHP との比較**
> - PHP 8 の名前付き引数 `json_encode(value: $x, flags: ...)` と同じ考え方です。
> - JS には名前付き引数が無いので、オブジェクト `{ maxRetries: 3, timeout: 60 }` を渡して代用します。
>   Python では言語の機能として持っています。

### キーワード専用引数 `*`（参考）

```python
def embed(self, texts, *, model, input_type):   # * より後ろはキーワードでしか渡せない
    ...
embed(texts, model="voyage-3.5", input_type="query")   # OK
embed(texts, "voyage-3.5", "query")                     # TypeError
```

ライブラリの関数でよく見かけます。「間違った順番で渡す事故」を構文で防ぐ書き方です。

### 落とし穴: 既定値に変更できるものを置かない

```python
def add(x, items=[]):     # ダメな例
    items.append(x)
    return items

>>> add(1)
[1]
>>> add(2)
[1, 2]                     # 前回の呼び出しの値が残っている！
```

**既定値は関数を定義したときに 1 回だけ作られ、呼び出しのたびに使い回されます。**
リストや辞書を既定値にすると、呼び出しをまたいで中身が残ります。Python の有名な罠です。

```python
def add(x, items=None):   # 正しい書き方
    if items is None:
        items = []
    items.append(x)
    return items
```

題材の既定値が `DEFAULT_MODEL`（文字列）や `256`（数値）など **変更できないものだけ** なのはこのためです。

---

## 4.3 複数の値を返す（タプルを返す）

### 題材コード

`scripts/build_kb.py:22-29`

```python
def parse_page(path: Path) -> tuple[dict[str, str], str]:
    """ページ先頭の <!-- key: value --> をメタ情報として取り出し、本文からは取り除く。"""
    text = path.read_text(encoding="utf-8")
    meta = dict(re.findall(r"<!--\s*([a-z_]+):\s*(.*?)\s*-->", text))
    body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return meta, body
```

`scripts/build_kb.py:92`

```python
meta, body = parse_page(path)
```

### 解説

`return meta, body` は **タプル `(meta, body)` を 1 個返している** のと同じです（カッコが省略されている）。
受け取る側でアンパックすると、複数の戻り値のように使えます。

```python
>>> result = parse_page(path)
>>> type(result)
<class 'tuple'>
>>> meta, body = result          # これを 1 行で書いたのが meta, body = parse_page(path)
```

戻り値の一部だけ使うときは添字で取り出すこともあります。

`reference/okubo-shamisen/eval.py:341`

```python
neg_t0 = [best_faq_match_t0(q, faq)[1] for q, _ in negatives]
```

`best_faq_match_t0()` は `(FAQのid, スコア)` を返すので、`[1]` でスコアだけを取り出しています。

> **JS / PHP との比較**
> JS は `return [meta, body]` + `const [meta, body] = ...`、PHP は `return [$meta, $body]` + `[$meta, $body] = ...`。
> 発想は同じですが、Python ではこれが最も自然な書き方で、標準ライブラリにも多数あります。
>
> **3 個を超えたら** タプルではなく辞書や dataclass（第 15 章）で返す方が、受け取る側で取り違えが起きません。
> `decide()` が 3 個のタプルを返しているのは、ちょうどその境目です。

### 早期 return（ガード節）

`reference/okubo-shamisen/eval.py:123-143`

```python
def decide(question, qvec, faq, cfg, router):
    got0, s0 = best_faq_match_t0(question, faq)
    if s0 >= cfg["tau0"]:
        return "t0", got0, []
    ranked = rank_faq_t1(qvec, faq)
    if ranked and ranked[0][1] >= cfg["tau1"]:
        return "t1", ranked[0][0], []
    cand_ids = [i for i, sc in ranked[: cfg["router_k"]] if sc >= cfg["tau1_cand"]]
    if not cand_ids:
        return "none", None, []
    if router is None:
        return "pending", None, cand_ids
    by_id = {e["id"]: e for e in faq}
    picked = router(question, [by_id[i] for i in cand_ids])
    return ("t1r", picked, cand_ids) if picked else ("none", None, cand_ids)
```

（型ヒントを省いて引用しています）

これは本番ボットの **4 段構成**（Tier0 → Tier1 → Tier1r）をそのまま 1 つの関数にしたものです。

1. 文字の一致度が閾値 τ0 以上なら、その FAQ で即答（`"t0"`）
2. 埋め込みの類似度の 1 位が τ1 以上なら、その FAQ で答える（`"t1"`）
3. 上位候補のうち下限以上のものを集める。1 つも無ければ定型回答なし（`"none"`）
4. 判定役（router）が無ければ「判定待ち」（`"pending"`）
5. 判定役に選ばせ、選ばれればその FAQ（`"t1r"`）、選ばれなければ `"none"`

**条件を満たしたらすぐ `return` する** ので、`if` が入れ子にならず、上から順に読めば流れが分かります。
これを **早期 return（ガード節）** と呼び、実務で強く推奨される書き方です。PHP / JS でも同じですね。

---

## 4.4 関数の中の関数とクロージャ

### 題材コード

`scripts/build_kb.py:52-85`

```python
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
```

### 解説

`chunk_page()` の中で、さらに `flush()` という関数を定義しています。

- `flush()` は、外側の関数の変数 `buf`・`chunks`・`heading`・`meta` を **引数で受け取らずに直接使っています**。
  内側の関数が外側の変数を覚えて使える仕組みを **クロージャ** と呼びます（JS と同じ）。
- 「見出しが来たとき」と「最後」の **2 か所で同じ処理が必要** なので、関数にまとめて重複を避けています。
- 外から呼ぶことはないので、`chunk_page()` の中に閉じ込めています（外の名前空間を汚さない）。

`heading` の値は、**`flush()` が呼ばれた時点の値** が使われます。
ループの中で `flush()` → `heading = 新しい見出し` の順なので、確定する塊には「切り替える前の見出し」が付きます。
これがちょうど「その塊が属していた見出し」になる、という巧妙な順序です。

### 落とし穴: 内側の関数で外の変数に代入できない

`flush()` が `buf = []` ではなく `buf.clear()` と書いているのには理由があります。

```python
def outer():
    buf = ["a"]
    def flush():
        text = "\n".join(buf)    # ← ここで UnboundLocalError
        buf = []
        return text
    return flush()
```

```
UnboundLocalError: cannot access local variable 'buf' where it is not associated with a value
```

Python は **関数の中のどこかで代入されている名前を、その関数のローカル変数とみなします**。
`buf = []` があるせいで、`flush()` 内の `buf` はすべて「flush のローカル変数」扱いになり、
まだ何も入っていない状態で `join` しようとしてエラーになります。

解決策は 2 つです。

| 方法 | 書き方 | 説明 |
|---|---|---|
| 中身を変更する（題材の方法） | `buf.clear()` | 変数への代入ではなく、指している **リストそのものを空にする** だけなので問題ない |
| `nonlocal` を宣言する | `nonlocal buf` を先頭に書いてから `buf = []` | 「外側の変数に代入する」と明示する |

同じ理由で、`chunks.append(...)` も「リストの中身の変更」なので問題なく動きます。

> **JS / PHP との比較**
> - JS は `let` で宣言した外の変数に、内側の関数からそのまま代入できます。Python は `nonlocal` が必要です。
> - PHP の無名関数は外の変数を `use ($buf)` で明示的に取り込み、書き換えるには `use (&$buf)` と参照にします。
>   Python は「読むのは自由、代入するには宣言」で、PHP と JS の中間くらいの感覚です。
> - モジュールのトップレベルの変数に関数の中から代入するには `global` を使いますが、実務ではほぼ使いません（設計を見直すサイン）。

### 関数の中の小さなヘルパー

`reference/okubo-shamisen/eval.py:378-385`

```python
def best_chunk(v: list[float]) -> float:
    return max(cosine(v, c["vec"]) for c in chunks)

pos_t2 = sorted(best_chunk(v) for v in pos_vecs)
neg_t2 = sorted(best_chunk(v) for v in neg_vecs)

def pct(xs: list[float], p: float) -> float:
    return xs[min(len(xs) - 1, int(p * len(xs)))]
```

`main()` の途中で、その場でしか使わない関数を定義しています。`best_chunk` は `main()` のローカル変数 `chunks` を
クロージャで使っています。こうした「その場限りの小関数」は長い `main()` を読みやすくする定番の手です。

`pct()` はパーセンタイル（小さい方から p の位置の値）を返します。`min(len(xs) - 1, ...)` は
`p = 1.0` のとき添字がはみ出さないようにするための上限です。

---

## 4.5 lambda（無名関数）と key 引数

### 題材コード

`reference/okubo-shamisen/eval.py:173, 196`

```python
return sorted(scored, key=lambda x: -x[1])
...
best = max(ok, key=lambda r: r["coverage"])
```

### 解説

`lambda 引数: 式` は、**1 つの式だけからなる小さな関数** をその場で作る書き方です。

```python
lambda x: -x[1]
# は、次とほぼ同じ
def _(x):
    return -x[1]
```

| | Python | JS | PHP |
|---|---|---|---|
| 書き方 | `lambda x: -x[1]` | `x => -x[1]` | `fn($x) => -$x[1]` |
| 中身 | **式 1 つだけ**（文は書けない） | 式 or ブロック | 式 1 つ |

`sorted()`・`max()`・`min()` は `key=` に関数を渡すと、**その関数の戻り値を基準に** 並べたり比べたりします。

- `scored` は `[("fee", 0.82), ("trial", 0.91), ...]` のような `(id, スコア)` のリスト
- `key=lambda x: -x[1]` は「スコアにマイナスを付けた値で並べる」= **スコアの大きい順**
- `sorted(scored, key=lambda x: x[1], reverse=True)` と書いても同じ結果です

`max(ok, key=lambda r: r["coverage"])` は「`coverage` が最大の辞書」を返します。
最大値そのもの（数値）ではなく **最大だった要素（辞書）** が返るのがポイントです。

`max()` は、同じ最大値が複数あると **先に出てきた方** を返します。`rows` は閾値 τ の小さい順に並んでいるので、
docstring の「同率なら低い方（＝カバー広い方）」はこの性質に頼っています。

> JS では `arr.sort((a, b) => b[1] - a[1])` のように **比較関数（2 引数）** を渡しますが、
> Python の `key=` は **1 引数で「比べる値」を返す関数** です。慣れると Python の方が書きやすく、間違いも減ります。

---

## 4.6 関数は「値」

Python の関数は、数値や文字列と同じく **変数に入れたり、引数で渡したり** できます。
`key=lambda ...` で関数を渡せるのもそのためです。

```python
>>> f = len
>>> f("三味線")
3
>>> sorted(["ccc", "a", "bb"], key=len)   # 組み込み関数をそのまま渡せる
['a', 'bb', 'ccc']
```

eval.py の `Router` クラスは、インスタンスを `router(question, cands)` と **関数のように呼べる** 作りです（第 10 章の `__call__`）。
`decide()` は `router` が関数なのかインスタンスなのかを気にせず、「呼べるもの」として扱っています。

---

## 4.7 可変長引数 `*args` / `**kwargs`（読めれば十分）

```python
def log(*args, **kwargs):
    print(args)     # 位置引数がタプルで入る
    print(kwargs)   # キーワード引数が辞書で入る

>>> log(1, 2, model="voyage", dim=256)
(1, 2)
{'model': 'voyage', 'dim': 256}
```

自分で書くことは少ないですが、ライブラリのコードやラッパー関数では頻繁に見かけます。
呼び出し側の `f(*xs)` / `f(**d)` は「リスト / 辞書を展開して引数として渡す」意味です（第 2 章の `*` `**` と同じ記号）。

---

## 章末問題

### 問 1（読解）

```python
def describe(model: str = "voyage-3.5-lite", dim: int = 256, input_type: str = "document") -> str:
    return f"{model}/{dim}/{input_type}"
```

次の呼び出しの結果は？ エラーになるものはどれ？

1. `describe()`
2. `describe("voyage-3.5", input_type="query")`
3. `describe(dim=1024)`
4. `describe(model="voyage-3.5", 1024)`

### 問 2（書く）

ページのリスト `pages = [("price", 1200), ("access", 340), ("faq", 5600)]`（キー, 文字数）を受け取り、
**文字数の最大と最小のページのキーを返す関数** `longest_and_shortest(pages)` を書いてください。
`max` / `min` の `key=` を使い、`("faq", "access")` を返すこと。

### 問 3（読解）

次のコードの出力は？

```python
def chunk(lines):
    out = []
    heading = ""
    def flush():
        out.append(heading)
    for line in lines:
        if line.startswith("#"):
            flush()
            heading = line
    flush()
    return out

print(chunk(["# A", "本文", "# B", "本文"]))
```

### 問 4（バグ探し）

次の関数を 2 回呼ぶと、2 回目に何が起きますか？ 直してください。

```python
def collect(text, seen=set()):
    for ch in text:
        seen.add(ch)
    return len(seen)

collect("ささら")   # 2
collect("しゃみ")   # ?
```

### 問 5（書く）

`scored = [("fee", 0.82), ("trial", 0.91), ("access", 0.40)]` を
**スコアの大きい順に並べ、0.5 以上の id だけのリスト** を返す関数 `top_ids(scored, threshold=0.5)` を書いてください。
結果: `['trial', 'fee']`

### 問 6（考える）

`decide()` は `(tier, id, 候補)` の 3 要素のタプルを返しています。
呼び出し側を読みやすくするために、どんな書き方が考えられますか？

---

## 回答

<details>
<summary>問 1 の回答</summary>

1. `'voyage-3.5-lite/256/document'`
2. `'voyage-3.5/256/query'`
3. `'voyage-3.5-lite/1024/document'`
4. **SyntaxError**（`positional argument follows keyword argument`）。キーワード引数の後ろに位置引数は書けません。

</details>

<details>
<summary>問 2 の回答</summary>

```python
def longest_and_shortest(pages: list[tuple[str, int]]) -> tuple[str, str]:
    longest = max(pages, key=lambda p: p[1])
    shortest = min(pages, key=lambda p: p[1])
    return longest[0], shortest[0]

print(longest_and_shortest([("price", 1200), ("access", 340), ("faq", 5600)]))
# ('faq', 'access')
```

</details>

<details>
<summary>問 3 の回答</summary>

`['', '# A', '# B']`

1 回目の `flush()` は最初の見出し `# A` の **前** に呼ばれるので、そのときの `heading` は `""`。
その後 `heading = "# A"` になり、`# B` の前の `flush()` では `# A`、最後の `flush()` では `# B` が入ります。
`heading = line` は `flush()` の中ではなく **外側の関数の中** での代入なので、`nonlocal` は不要です。
`flush()` は代入せず読むだけなので、呼ばれた時点の値が見えます。

</details>

<details>
<summary>問 4 の回答</summary>

2 回目は `5` になります（期待は 3）。既定値の `set()` は関数定義時に 1 回だけ作られ、
1 回目の `{"さ", "ら"}` が残ったまま `{"し", "ゃ", "み"}` が足されるためです。

```python
def collect(text, seen=None):
    if seen is None:
        seen = set()
    for ch in text:
        seen.add(ch)
    return len(seen)
```

</details>

<details>
<summary>問 5 の回答</summary>

```python
def top_ids(scored: list[tuple[str, float]], threshold: float = 0.5) -> list[str]:
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    result = []
    for id_, score in ranked:
        if score >= threshold:
            result.append(id_)
    return result
```

第 5 章の内包表記を使えば `return [i for i, s in ranked if s >= threshold]` の 1 行になります
（eval.py:136 の `cand_ids = [...]` がまさにこの形です）。
`id` は組み込み関数の名前なので、変数名には `id_` と末尾に `_` を付けるのが慣習です（export_kb.py も `id_`）。

</details>

<details>
<summary>問 6 の回答例</summary>

- **名前付きで返す**: `typing.NamedTuple` や `@dataclass`（第 15 章）で `Decision(tier=..., faq_id=..., candidates=...)` を返すと、
  呼び出し側で `d.tier` / `d.faq_id` と名前で読めます。アンパックもそのまま使えるので、既存の呼び出し側を壊しません。
- 辞書 `{"tier": ..., "id": ...}` で返す方法もありますが、キーの打ち間違いに気付けないので、型の付いた NamedTuple / dataclass の方が実務では好まれます。

```python
from typing import NamedTuple

class Decision(NamedTuple):
    tier: str
    faq_id: str | None
    candidates: list[str]

d = Decision("t1", "fee", [])
print(d.tier)          # 名前で読める
tier, got, cands = d   # タプルとしてアンパックもできる
```

</details>
