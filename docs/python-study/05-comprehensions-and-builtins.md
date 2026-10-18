# 第 5 章 内包表記・ジェネレータ式・組み込み関数

頻度: ★★★★☆ — 「Python らしいコード」の代表。読めないと、実務のコードの 2 割くらいが暗号に見えます。

この章で読めるようになるコード:

```python
return [p.strip() for p in parts if len(p.strip()) >= CHUNK_MIN]
by_id = {e["id"]: e for e in faq}
dot = sum(x * y for x, y in zip(a, b))
return rid if any(c["id"] == rid for c in cands) else None
SOFTWARE_NAMES = [n for e in json.loads(...) for n in [e["name"], *e.get("aliases", [])]]
```

---

## 5.1 リスト内包表記

### 題材コード

`scripts/build_kb.py:104`

```python
lens = [len(c["text"]) for c in chunks]
```

### 解説

**`[式 for 変数 in イテラブル]`** で、「イテラブルの各要素に式を適用した新しいリスト」を作ります。
次の for 文と同じ意味です。

```python
lens = []
for c in chunks:
    lens.append(len(c["text"]))
```

3 行が 1 行になり、しかも「`lens` は `chunks` の各テキストの長さのリスト」という **意図が一目で分かる** のが利点です。

> **JS / PHP との比較**
>
> ```javascript
> const lens = chunks.map(c => c.text.length);                     // JS
> ```
>
> ```php
> $lens = array_map(fn($c) => mb_strlen($c['text']), $chunks);    // PHP
> $lens = collect($chunks)->map(fn($c) => mb_strlen($c['text']));  // Laravel Collection
> ```
>
> Python にも `map()` / `filter()` はありますが、**内包表記の方が好まれます**。

Python の `len()` は **文字数** を数えます（PHP の `strlen` のようにバイト数にはならない）。
日本語でも `len("三味線") == 3` です。PHP の `mb_strlen` に当たります。

---

## 5.2 条件付きの内包表記（絞り込み）

### 題材コード

`scripts/build_kb.py:49`

```python
return [p.strip() for p in parts if len(p.strip()) >= CHUNK_MIN]
```

`reference/okubo-shamisen/eval.py:136`

```python
cand_ids = [i for i, sc in ranked[: cfg["router_k"]] if sc >= cfg["tau1_cand"]]
```

### 解説

**`[式 for 変数 in イテラブル if 条件]`** — 条件を満たす要素だけを、式で変換して集めます。
JS の `.filter(...).map(...)` を 1 つにしたものです。

`cand_ids` の行を分解して読むと:

1. `ranked[: cfg["router_k"]]` — スコア順に並んだ `(id, スコア)` の上位 k 件（本番設定では 3 件）
2. `for i, sc in ...` — それぞれを id とスコアにアンパック
3. `if sc >= cfg["tau1_cand"]` — スコアが候補の下限（0.60）以上のものだけ
4. `i` — id だけを集める

→ 「上位 3 件のうち類似度 0.60 以上の FAQ の id」。本番ボットの「判定役の LLM に渡す候補」です。

### 読む順番のコツ

内包表記は **`for` から読み始め、`if` を読み、最後に先頭の式に戻る** と読みやすくなります。

```
[ p.strip()          for p in parts          if len(p.strip()) >= CHUNK_MIN ]
   ③ これを集める     ① parts の各 p について   ② 短すぎなければ
```

### if-else を使う場合（位置に注意）

**変換の途中で分岐する** 場合は、条件式（第 3 章）を **先頭の式の中に** 書きます。

```python
[x if x >= 0 else 0 for x in xs]     # 負の数を 0 にする（全要素が残る）
[x for x in xs if x >= 0]            # 負の数を捨てる（要素が減る）
```

`for` の後ろの `if` は「絞り込み」、先頭の `if ... else` は「値の選択」です。混同しやすいので注意。

---

## 5.3 辞書内包表記とセット内包表記

### 題材コード

`reference/okubo-shamisen/eval.py:141, 248`

```python
by_id = {e["id"]: e for e in faq}
...
self_hits = {tau: 0 for tau in (0.70, 0.80, 0.85)}
```

### 解説

**`{キー: 値 for 変数 in イテラブル}`** で辞書を作れます。

- `by_id` は、FAQ のリストから **「id → FAQ」の索引（辞書）** を作っています。
  リストのまま `id` で探すと毎回先頭から順に見ることになりますが、辞書にしておけば一発で引けます。
  直後の `[by_id[i] for i in cand_ids]` で、候補 id から FAQ 本体を取り出しています。
- `self_hits` は、閾値ごとのヒット数を 0 で初期化しています。

PHP なら `array_column($faq, null, 'id')`、Laravel なら `collect($faq)->keyBy('id')` に当たる操作です。
実務で **非常によく使う** ので覚えておいてください。

`{式 for ...}` と `:` が無ければ **セット内包表記** です（重複を除いて集める）。

```python
departments = {row["部門"] for row in rows}   # 部門の種類（重複なし）
```

---

## 5.4 入れ子の内包表記（for を 2 つ以上）

### 題材コード

`reference/okubo-shamisen/eval.py:269`

```python
pipe_cases = [(v, e["id"]) for e in faq for q in e["questions"] for v in light_variants(q)]
```

`PivotAggregationSystem/scripts/generate_dummy_csv.py:169`

```python
SOFTWARE_NAMES = [n for e in json.loads(SOFTWARE_MASTER_PATH.read_text(encoding="utf-8")) for n in [e["name"], *e.get("aliases", [])]]
```

### 解説

**`for` を複数並べると、左から順に入れ子の for 文** になります。`pipe_cases` は次と同じです。

```python
pipe_cases = []
for e in faq:                          # FAQ エントリごとに
    for q in e["questions"]:           # その質問文ごとに
        for v in light_variants(q):    # 前置き・後置きの変形ごとに
            pipe_cases.append((v, e["id"]))
```

「全 FAQ の、全質問文の、3 通りの変形」を `(質問, 正解のid)` の組で並べた **テストケースの一覧** です。

`SOFTWARE_NAMES` も同じ形で、

```python
SOFTWARE_NAMES = []
for e in json.loads(...):                         # マスタの各ソフトについて
    for n in [e["name"], *e.get("aliases", [])]:  # 正式名と、別名すべて
        SOFTWARE_NAMES.append(n)
```

つまり **「入れ子のリストを平らにする（flatten）」** 処理です。JS の `flatMap`、Laravel の `flatMap` に当たります。

> **読みやすさの目安**
> for が 2 つまでなら内包表記、3 つ以上や条件が複雑なら普通の for 文で書く、という感覚が一般的です。
> `SOFTWARE_NAMES` の 1 行は 150 文字を超えていて、実務のレビューでは「分けた方が読みやすい」と言われる可能性があります。
> 内包表記は **短く書くための道具ではなく、意図を明確にするための道具** です。

---

## 5.5 ジェネレータ式 — カッコが `()` になると何が違うか

### 題材コード

`reference/okubo-shamisen/eval.py:151-155`

```python
def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
```

### 解説

`sum(x * y for x, y in zip(a, b))` の中身は `[ ]` が無い内包表記です。これを **ジェネレータ式** と呼びます。

| | リスト内包表記 `[...]` | ジェネレータ式 `(...)` |
|---|---|---|
| 結果 | リストを **全部作ってメモリに置く** | 値を **1 個ずつその場で作る** |
| 何度も使えるか | 使える | **1 回きり**（読み切ったら空） |
| 用途 | 結果を保存・何度も使う | `sum` / `max` / `any` などに **そのまま渡して 1 回で使い切る** |

関数の引数にするときは、`sum((x * y for ...))` の外側のカッコを省略して `sum(x * y for ...)` と書けます。

```python
>>> g = (x * x for x in range(3))
>>> list(g)
[0, 1, 4]
>>> list(g)           # 2 回目は空！
[]
```

256 次元のベクトルなら差はわずかですが、100 万件だとリストは数 MB のメモリを使い、
ジェネレータはほぼ使いません。「**合計・最大・判定に渡すだけならジェネレータ式**」と覚えてください。

### cosine を読む — コサイン類似度

この関数は RAG の検索の心臓部、**コサイン類似度** の計算です。

- `dot` — 内積。2 つのベクトルの同じ位置どうしを掛けて全部足す
- `na`, `nb` — それぞれのベクトルの長さ（各要素の 2 乗の和の平方根）
- `dot / (na * nb)` — 向きがどれだけ揃っているか（1 に近いほど似ている）
- `if na and nb else 0.0` — 長さ 0 のベクトルで割り算エラーにならないようにする

数式で書くと `cos θ = (a・b) / (|a| |b|)` です。これが 3 行で書けるのが内包表記・ジェネレータ式の力です。

（実務で大量に計算するなら NumPy を使います。`load_kb.py` が `np.array(v)` を使っているのは pgvector との受け渡しのためです。）

---

## 5.6 組み込み関数と組み合わせる

題材に出てくる、ジェネレータ式と相性の良い組み込み関数です。

### sum — 数える・合計する

`reference/okubo-shamisen/eval.py:181-182`

```python
covered = sum(1 for ok, s in results_pos if ok and s >= tau)
false_ans = sum(1 for s in results_neg if s >= tau)
```

`sum(1 for ... if 条件)` は **「条件を満たす件数を数える」** 定番イディオムです。
条件を満たすたびに 1 を出し、それを合計しています。

`generate_dummy_csv.py` の案件の合計計算も同じ形です。

```python
cost_total = sum(ln["仕入小計(税別)"] for ln in lines)
```

### max / min — key= との組み合わせ

`reference/okubo-shamisen/eval.py:170`

```python
(e["id"], max(cosine(qvec, v) for v in e["qvecs"]))
```

FAQ 1 件には質問文の言い換えが複数あり、それぞれのベクトル（`qvecs`）を持っています。
「質問と、そのうち **いちばん近い言い換え** の類似度」を FAQ のスコアにしています。

### any / all — 1 つでも / すべて

`reference/okubo-shamisen/eval.py:120`

```python
return rid if any(c["id"] == rid for c in cands) else None
```

`generate_dummy_csv.py:435`

```python
"送料": "込み" if all(ln["商品フラグ"] == 1 for ln in lines) else "別途",
```

- `any(...)` — 1 つでも真なら True（見つかった時点で打ち切る）。JS の `.some()`、Laravel の `contains()`
- `all(...)` — すべて真なら True（偽が出た時点で打ち切る）。JS の `.every()`

eval.py は「LLM が返した id が、渡した候補のどれかと一致するか」を確認し、
候補に無い id（LLM のでっち上げ）を `None` にしています。**LLM の出力を信用しすぎない** ための検証です。

> **空のときの値に注意**: `any([])` は `False`、`all([])` は **`True`** です（「反例が 1 つも無い」から）。
> dummy_csv では、明細が 1 行も無い案件は `all(...)` が `True` になり、送料は「込み」になります。
> 意図どおりか、境界の値を考える習慣を付けましょう。

### "".join(ジェネレータ式)

`reference/okubo-shamisen/eval.py:114`

```python
reply = "".join(b.text for b in msg.content if b.type == "text")
```

Claude API の応答 `msg.content` は「テキスト」「ツール呼び出し」などのブロックのリストです。
テキストのブロックだけを取り出してつなげています。`join` にもジェネレータ式をそのまま渡せます。

### sorted — どんなイテラブルも並べ替えたリストにする

```python
pos_t2 = sorted(best_chunk(v) for v in pos_vecs)        # eval.py:381
for path in sorted(PAGES_DIR.glob("*.md")):             # build_kb.py:91
```

- `sorted(x)` は **新しいリスト** を返します（元はそのまま）。ジェネレータでも何でも受け取れます。
- `xs.sort()` は **リスト自身を並べ替え**、戻り値は `None` です。`ys = xs.sort()` と書くと `ys` が `None` になる、という罠があります。
- `glob()` が返すファイルの順番は OS 次第なので、build_kb は `sorted()` で **毎回同じ順番** にしています。
  チャンクの通し番号 `c0000, c0001, ...` が実行のたびに変わらないための工夫です。

### 組み込み関数の早見表

| 関数 | 意味 | 題材 |
|---|---|---|
| `len(x)` | 要素数・文字数 | 全ファイル |
| `sum(it)` | 合計 | eval, dummy_csv |
| `max(it, key=)` / `min(...)` | 最大・最小（key で基準を指定） | eval |
| `sorted(it, key=, reverse=)` | 並べ替えた新しいリスト | eval, build_kb |
| `any(it)` / `all(it)` | 1 つでも / すべて | eval, dummy_csv |
| `enumerate(it)` | (番号, 要素) | build_kb, embedders |
| `zip(a, b)` | 同じ位置の組 | 多数 |
| `range(...)` | 整数の並び | 多数 |
| `round(x, n)` | 丸め（銀行丸め） | export_kb |
| `int()` `float()` `str()` `bool()` | 型の変換 | dummy_csv |
| `list()` `dict()` `set()` `tuple()` | 入れ物の変換・作成 | build_kb |
| `isinstance(x, 型)` | 型の判定 | （題材には無いが頻出） |
| `print()` | 出力 | 全ファイル |

---

## 5.7 内包表記で辞書のリストを作る

### 題材コード

`scripts/export_kb.py:50-60`

```python
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
```

### 解説

DB から取った **タプルの行** を、JSON に書き出せる **辞書の行** に変換しています。

- `for id_, page_key, title, heading, text, emb in rows` — 1 行（6 列のタプル）をアンパック
- 先頭の式は **辞書リテラル** で、複数行にまたがっていても内包表記の「式」の部分です
- `"vec"` の値の中に **さらに内包表記** があり、ベクトルの各要素を小数 4 桁に丸めています

「DB の行 → API / JSON の形」の変換は、実務の Python で最もよく書くコードの 1 つです。
PHP の Laravel で言うと Eloquent の結果を `->map(fn($r) => [...])` で整形するのと同じ作業です。

---

## 章末問題

### 問 1（書き換え）次の for 文を内包表記 1 行に書き換えてください

```python
ids = []
for c in chunks:
    if c["heading"]:
        ids.append(c["id"])
```

### 問 2（読解）次の値は？

```python
xs = [3, -1, 4, -1, 5]
a = [x for x in xs if x > 0]
b = [x if x > 0 else 0 for x in xs]
c = {x for x in xs}
d = sum(1 for x in xs if x < 0)
e = {str(x): x * x for x in xs if x > 3}
```

### 問 3（読解）

`taus = [round(0.30 + 0.05 * i, 2) for i in range(13)]`（eval.py:239）は、どんなリストになりますか？
`round(..., 2)` が無いと何が困りますか？（ヒント: REPL で `0.30 + 0.05 * 6` を打ってみる）

### 問 4（書く）

次の FAQ のリストから、`{"質問文": "FAQのid"}` の辞書を **内包表記 1 つで** 作ってください（入れ子の for を使う）。

```python
faq = [
    {"id": "fee", "questions": ["月謝は？", "料金は？"]},
    {"id": "trial", "questions": ["体験できる？"]},
]
# → {'月謝は？': 'fee', '料金は？': 'fee', '体験できる？': 'trial'}
```

### 問 5（書く）

`cosine()` と同じ考え方で、**2 つのベクトルのユークリッド距離**（同じ位置の差の 2 乗を全部足して平方根）を
ジェネレータ式で 1 行で返す関数 `distance(a, b)` を書いてください。`distance([0, 0], [3, 4])` は `5.0`。

### 問 6（考える）

次のコードの問題点は？

```python
scores = (cosine(q, c["vec"]) for c in chunks)
print("最大:", max(scores))
print("平均:", sum(scores) / len(chunks))
```

---

## 回答

<details>
<summary>問 1 の回答</summary>

```python
ids = [c["id"] for c in chunks if c["heading"]]
```

</details>

<details>
<summary>問 2 の回答</summary>

- `a` = `[3, 4, 5]`（絞り込み）
- `b` = `[3, 0, 4, 0, 5]`（値の選択。要素数は変わらない）
- `c` = `{3, -1, 4, 5}`（セット。重複が消える。表示順は保証されない）
- `d` = `2`（負の数の個数）
- `e` = `{'4': 16, '5': 25}`

</details>

<details>
<summary>問 3 の回答</summary>

`[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]`（0.30 から 0.05 刻みで 13 個）。

`round` が無いと、小数（浮動小数点数）の誤差で `0.6000000000000001` や `0.8500000000000001` のような値が混ざります。
レポートの表示が崩れるだけでなく、`tau == 0.85` のような比較が一致しなくなります（eval.py の `self_hits` は `0.85` をキーにしています）。
**小数を足し算で作るときは誤差が出る** ことを覚えておきましょう（JS でも `0.1 + 0.2 !== 0.3` で同じです）。

</details>

<details>
<summary>問 4 の回答</summary>

```python
q_to_id = {q: e["id"] for e in faq for q in e["questions"]}
```

for の順番は「外側（FAQ）→ 内側（その質問文）」です。逆に書くと `e` が未定義でエラーになります。

</details>

<details>
<summary>問 5 の回答</summary>

```python
def distance(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

print(distance([0, 0], [3, 4]))  # 5.0
```

</details>

<details>
<summary>問 6 の回答</summary>

`scores` はジェネレータなので **1 回しか読めません**。`max(scores)` で読み切った後の `sum(scores)` は `0` になり、
平均が 0 と表示されます（エラーにならないのが厄介）。
何度も使うなら `[ ]` でリストにします。

```python
scores = [cosine(q, c["vec"]) for c in chunks]
print("最大:", max(scores))
print("平均:", sum(scores) / len(scores))
```

</details>
