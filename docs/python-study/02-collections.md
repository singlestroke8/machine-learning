# 第 2 章 リスト・辞書・タプル・セット

頻度: ★★★★★ — データを「まとめて持つ」入れ物です。実務のコードの半分はこれらの出し入れです。

PHP は「配列（array）」1 つで、順番付きのリストも、キー付きの連想配列もこなします。
Python では役割ごとに **4 種類の入れ物** を使い分けます。ここが PHP 出身者の最初の壁です。

| 入れ物 | 書き方 | 順序 | 変更 | 主な用途 | PHP / JS で言うと |
|---|---|---|---|---|---|
| `list` リスト | `[1, 2, 3]` | あり | できる | 同じ種類のものを並べる | 添字配列 / `Array` |
| `dict` 辞書 | `{"id": "c0001"}` | 挿入順 | できる | キーで値を引く | 連想配列 / `Object`・`Map` |
| `tuple` タプル | `("query", 0.92)` | あり | **できない** | 決まった個数の組 | — （JS の `[a, b]` で代用） |
| `set` セット | `{"朝日", "商品部"}` | なし | できる | 重複なし・所属判定 | — / `Set` |

---

## 2.1 リスト（list）

### 題材コード

`scripts/embedders.py:50-62`（抜粋）

```python
out: list[list[float] | None] = [None] * len(texts)
misses: list[int] = []
for i, t in enumerate(texts):
    p = self._cache_path(t, input_type)
    if p.exists():
        out[i] = json.loads(p.read_text())
        self.cache_hits += 1
    else:
        misses.append(i)

for start in range(0, len(misses), BATCH):
    idx = misses[start : start + BATCH]
```

### 解説

このコードは「キャッシュに無かったものの **位置（添字）** だけを `misses` に控え、後で API に投げる」処理です。
リストの基本操作がまとまって出てきます。

| 操作 | 書き方 | 意味 |
|---|---|---|
| 空のリスト | `misses = []` | |
| 同じ値で n 個 | `[None] * len(texts)` | 先に「席」を用意しておき、後で `out[i] = ...` で埋める |
| 末尾に追加 | `misses.append(i)` | PHP の `$misses[] = $i;`、JS の `push` |
| 添字で読み書き | `out[i] = vec` | 0 始まり |
| 長さ | `len(misses)` | PHP の `count()`、JS の `.length` |
| 切り出し（スライス） | `misses[start : start + BATCH]` | 下で詳しく |

`[None] * len(texts)` で **入力と同じ長さの席** を最初に作っているのがポイントです。
キャッシュから読めたもの・API から返ったものを、それぞれ元の位置 `out[i]` に戻すので、
**出力の順番が入力と必ず一致** します。

### スライス

`リスト[開始:終了]` で一部を切り出した **新しいリスト** が得られます。終了の位置は含みません。

```python
>>> xs = ["a", "b", "c", "d", "e"]
>>> xs[1:3]      # 1 番目から 3 番目の手前まで
['b', 'c']
>>> xs[:3]       # 先頭から 3 個（eval.py の ranked[:3] = 上位 3 件）
['a', 'b', 'c']
>>> xs[3:]       # 3 番目から最後まで
['d', 'e']
>>> xs[-1]       # 最後の要素（マイナスは後ろから数える）
'e'
>>> xs[10:20]    # 範囲外でもエラーにならず、空リスト
[]
>>> xs[10]       # 1 個だけ取り出すときは範囲外でエラー
IndexError: list index out of range
```

`misses[start : start + BATCH]` は、`range(0, len(misses), BATCH)` で `start` を 0, 128, 256, ... と進めながら、
**128 件ずつ取り出す** 定番の書き方です（バッチ処理）。最後の回は 128 件に満たなくてもエラーになりません。

マイナスの添字は eval.py で、並べ替えた後の最大値を取るのに使われています。

`reference/okubo-shamisen/eval.py:390`

```python
f"| 正答 | {pos_t2[0]:.2f} | ... | {pos_t2[-1]:.2f} |",
```

`sorted()` で小さい順に並べたので、`[0]` が最小、`[-1]` が最大です。

> **JS / PHP との比較**
> JS の `arr.slice(1, 3)` / PHP の `array_slice($a, 1, 2)` に当たります。
> PHP の第 3 引数は「個数」、Python は「終了位置」なので注意。
> 後ろからの添字 `xs[-1]` は JS では `xs.at(-1)`、PHP では `end($xs)` などで書きます。

### よく使うリストのメソッド

| メソッド | 意味 | 題材 |
|---|---|---|
| `xs.append(x)` | 末尾に 1 個追加 | 多数 |
| `xs.extend(ys)` | 末尾に **リストの中身を全部** 追加 | build_kb の `chunks.extend(chunk_page(...))` |
| `xs.insert(i, x)` | i 番目に挿入 | dummy_csv の `names.insert(...)` |
| `xs.clear()` | 空にする | build_kb の `buf.clear()` |
| `xs.sort()` | その場で並べ替え（戻り値は None） | |
| `x in xs` | 含まれるか | |

`append` と `extend` の違いはよく問われます。

```python
>>> chunks = [1, 2]
>>> chunks.append([3, 4])   # リストを「1 個の要素」として追加
>>> chunks
[1, 2, [3, 4]]
>>> chunks = [1, 2]
>>> chunks.extend([3, 4])   # 中身をばらして追加（PHP の array_merge、JS の push(...arr)）
>>> chunks
[1, 2, 3, 4]
```

build_kb の `main()` は、ページごとに `chunk_page()` が返すリストを `extend` でつないで、
**全ページ分のチャンクを 1 本のリスト** にしています。

---

## 2.2 辞書（dict）

### 題材コード

`reference/okubo-shamisen/textmatch.py:20-28`

```python
def bigrams(text: str) -> dict[str, int]:
    """正規化済みテキストの文字バイグラム頻度。1文字の場合はその文字自身。"""
    if len(text) < 2:
        return {text: 1} if text else {}
    grams: dict[str, int] = {}
    for i in range(len(text) - 1):
        g = text[i : i + 2]
        grams[g] = grams.get(g, 0) + 1
    return grams
```

### 解説

「文字バイグラム」は、文章を **隣り合う 2 文字ずつ** に区切ったものです。
`"体験レッスン"` なら `体験 / 験レ / レッ / ッス / スン` の 5 つ。それぞれ何回出たかを辞書に数えています。

```python
>>> bigrams("ののの")
{'のの': 2}
```

- `{}` が空の辞書。`{キー: 値, ...}` で作ります。
- `grams[g] = ...` で代入（キーが無ければ新しく作られる）。
- **`grams.get(g, 0)`** は「キー `g` があればその値、無ければ `0`」。
  `grams[g]` と書くと、無いキーのときに **`KeyError` で止まります**。
- 文字列もスライスできます。`text[i : i + 2]` は i 文字目から 2 文字。

> **PHP との比較**
> PHP で同じことを書くと `$grams[$g] = ($grams[$g] ?? 0) + 1;` です。
> Python の `d.get(key, 既定値)` が PHP の `$d[$key] ?? 既定値` に当たると覚えると早いです。
> なお、PHP の未定義キーは Warning で済みますが、Python の `d[key]` は **例外で処理が止まります**。

### `d[key]` と `d.get(key)` の使い分け — 本番版と再実装版の比較

同じ処理が、本番ボット版と再実装版で **わざと違う書き方** になっています。

`reference/okubo-shamisen/build_kb.py`（本番ボット版）

```python
"page_key": meta.get("page_key", ""),
"title": meta.get("title", ""),
```

`scripts/build_kb.py:66-69`（再実装版）

```python
# meta["..."] は、コメント行が欠けたページがあれば KeyError で止まる。
# 空文字のまま DB の NOT NULL 列に入るのを防ぐため
"page_key": meta["page_key"],
"title": meta["title"],
```

- `.get(key, "")` は **エラーにせず空文字で続行** します。
- `[key]` は **無ければその場で止まります**。

再実装版は、ページのメタ情報が欠けていたら「空文字のまま DB に入る」より「すぐ止まって気付ける」方を選んでいます。
これを **フェイルファスト（早く失敗させる）** と言います。
**無いことがあり得て、既定値で困らない → `.get()`**、**必ずあるはずで、無ければバグ → `[]`** が実務での使い分けです。

### 辞書のループ

```python
dot = sum(v * gb.get(k, 0) for k, v in ga.items())        # textmatch.py:36
na = sum(v * v for v in ga.values()) ** 0.5               # textmatch.py:39
```

| 書き方 | 取り出せるもの | PHP |
|---|---|---|
| `for k in d:` | キー | `foreach (array_keys($d) as $k)` |
| `for v in d.values():` | 値 | `foreach ($d as $v)` |
| `for k, v in d.items():` | キーと値の組 | `foreach ($d as $k => $v)` |

**`for x in d` はキーを回す** 点に注意してください（PHP の `foreach ($d as $x)` は値を回します）。

### 辞書の作り方いろいろ

`scripts/build_kb.py:26`

```python
meta = dict(re.findall(r"<!--\s*([a-z_]+):\s*(.*?)\s*-->", text))
```

`re.findall` は `[("page_key", "price"), ("title", "料金")]` のような **「2 要素の組」のリスト** を返します。
`dict()` に「キーと値の組のリスト」を渡すと、それを辞書に変換してくれます。

```python
>>> dict([("page_key", "price"), ("title", "料金")])
{'page_key': 'price', 'title': '料金'}
```

正規表現は第 11 章で扱います。ここでは「組のリスト → `dict()` で辞書」という変換を覚えてください。

### 辞書の合成 `{**a, **b}`

`generate_dummy_csv.py:488`

```python
merged = {**order, **detail}
```

`**辞書` は「辞書の中身をここに展開する」記号です。`order`（案件の列）と `detail`（明細の列）を 1 つの辞書にまとめています。
**同じキーがあれば後ろが勝ちます。**

```python
>>> order = {"品名": "", "顧客名": "ダミー商事"}
>>> detail = {"品名": "ThinkPad X280"}
>>> {**order, **detail}
{'品名': 'ThinkPad X280', '顧客名': 'ダミー商事'}
```

JS のスプレッド `{...order, ...detail}` とまったく同じ感覚です。
PHP では `array_merge($order, $detail)` が近いですが、数値キーの扱いが違うので PHP の方が罠が多いです。
Python 3.9 以降は `order | detail` とも書けます。

辞書リテラルの中で **条件付きで項目を足す** 使い方もあります。

`generate_dummy_csv.py:288`

```python
lines.append({
    "品名": name,
    ...
    **(make_software(rng) if rng.random() < SOFTWARE_LINE_RATE else {}),
})
```

3 割の確率で `make_software()` が返す `{"ソフト": ..., "ソフト数": ...}` を展開し、
それ以外は空の辞書 `{}` を展開（= 何も足さない）しています。

### 辞書の順序

Python 3.7 以降、辞書は **入れた順番を保ちます**。
`json.dumps()` で書き出したときに列の順番が毎回同じになるのはこのおかげです。

---

## 2.3 タプル（tuple）

### 題材コード

`scripts/load_kb.py:51`

```python
deleted = conn.execute("DELETE FROM chunks WHERE model = %s", (emb.model,)).rowcount
```

### 解説

- タプルは `( )` で作る、**後から変更できないリスト** です。
- 「決まった個数で、それぞれ意味が違う値の組」に使います。例: `(質問, 期待するid)`、`(列名, 重み)`。
- 変更できないので、辞書のキーやセットの要素にも使えます（リストはできません）。

### 要素 1 個のタプルには `,` が必須

`(emb.model,)` の **最後のカンマが重要** です。

```python
>>> type(("voyage-3.5"))      # カッコはただの計算の優先順位を表すだけ
<class 'str'>
>>> type(("voyage-3.5",))     # カンマがあって初めてタプル
<class 'tuple'>
```

DB ライブラリの `execute(SQL, パラメータ)` は、パラメータを **タプル（またはリスト）** で受け取ります。
カンマを忘れて `("voyage-3.5")` を渡すと、ただの文字列になってしまいます。
psycopg（3 系）は「パラメータは文字列ではなくシーケンスで渡して」という `TypeError` で止めてくれますが、
古いライブラリ（psycopg2 など）では文字列を「1 文字ずつの値の並び」と解釈して、分かりにくいエラーになります。
**実務で本当によく踏む罠** です。

### アンパック（分割代入）

タプル（やリスト）は、複数の変数に一度に代入できます。

`generate_dummy_csv.py:258`

```python
name, cost, (lo, hi) = rng.choice(PRODUCTS)
```

`PRODUCTS` の各要素は `("Let's note SV7 青", 19980, (1.8, 2.5))` という形です。
**入れ子の形をそのまま左辺に書く** と、内側のタプルまで一度にばらせます。

```python
>>> name, cost, (lo, hi) = ("ThinkPad X280", 24800, (1.6, 2.1))
>>> lo, hi
(1.6, 2.1)
```

使わない要素は `_` で受けるのが慣習です。

`generate_dummy_csv.py:308`

```python
status, status_group, _ = rng.choices(STATUSES, weights=[s[2] for s in STATUSES])[0]
```

`STATUSES` の各要素は `(ステータス名, まとめ, 出現の重み)`。重みは選ぶときにしか使わないので `_` で捨てています。

> **JS / PHP との比較**
> JS の分割代入 `const [name, cost, [lo, hi]] = ...`、PHP の `[$name, $cost, [$lo, $hi]] = ...` と同じです。
> Python は左辺の `[ ]` が不要で、カンマで並べるだけです。

`a, b = b, a` で 2 つの変数を入れ替えられるのも、このアンパックの応用です。

### `*` で展開する

`generate_dummy_csv.py:169`（一部）

```python
for n in [e["name"], *e.get("aliases", [])]
```

`*リスト` はリストの中身をその場に並べます（JS の `...arr`）。
`[e["name"], *e["aliases"]]` で「正式名 + 別名すべて」を 1 本のリストにしています。
`e.get("aliases", [])` で、別名が無いソフトでも空リストを展開する（= 何も足さない）ようにしています。

---

## 2.4 セット（set）

### 題材コード

`generate_dummy_csv.py:129, 316`

```python
PP_DEPARTMENTS = {"プロジェクトパートナー部", "アウトソース部", "朝日"}
...
if department in PP_DEPARTMENTS:
```

### 解説

- `{値, 値, ...}` で作る、**重複のない・順序のない** 集まりです。
- 主な用途は **「含まれるか」の判定** と **重複の除去**。
- `x in セット` は要素数が増えても速いです（ハッシュで探す）。`x in リスト` は先頭から順に探すので、
  要素が数千を超えるとはっきり遅くなります。

```python
>>> set(["振込", "振込", "代引き"])     # 重複の除去
{'振込', '代引き'}
>>> {"a", "b"} & {"b", "c"}             # 共通部分
{'b'}
>>> {"a", "b"} | {"b", "c"}             # 和集合
{'a', 'b', 'c'}
>>> {"a", "b"} - {"b"}                  # 差
{'a'}
```

> **注意:** 空のセットは `set()` です。**`{}` は空の辞書** になります。

---

## 2.5 変更できる / できない と「参照」

リスト・辞書・セットは **変更できる（mutable）**、数値・文字列・タプル・None は **変更できない（immutable）** です。

変更できるものを変数に代入すると、**コピーではなく同じ実体を指す** だけです（JS のオブジェクト・配列と同じ）。

```python
>>> a = [1, 2]
>>> b = a          # コピーではない
>>> b.append(3)
>>> a
[1, 2, 3]          # a も変わる
>>> c = a.copy()   # コピーしたいときは明示する（a[:] や list(a) でも可）
```

PHP の配列は代入で **コピー** される（値渡し）ので、PHP 出身者がもっとも驚くところです。
JS の配列・オブジェクトと同じ、と考えてください。

### `[[]] * n` の罠

```python
>>> rows = [[]] * 3
>>> rows[0].append("x")
>>> rows
[['x'], ['x'], ['x']]    # 3 つとも同じリストを指している！
```

`[None] * n`（embedders.py）は None が変更できないので安全ですが、
**変更できるもの（リスト・辞書）を `*` で増やすと全部同じ実体** になります。
そういうときは内包表記（第 5 章）で `[[] for _ in range(3)]` と書きます。

---

## 2.6 落とし穴まとめ

| 落とし穴 | 対策 |
|---|---|
| `d[key]` で KeyError | 無いことがあり得るなら `d.get(key, 既定値)` |
| `for x in d` で値が来ると思った | キーが来る。値なら `.values()`、両方なら `.items()` |
| 1 要素タプルのカンマ忘れ `(x)` | `(x,)` |
| 空セットを `{}` で書く | `set()` |
| 代入でコピーされたと思った | `.copy()` / `list(xs)` / `dict(d)`（入れ子まで複製するなら `copy.deepcopy`） |
| `[[]] * n` | `[[] for _ in range(n)]` |
| `append` と `extend` の取り違え | 中身をばらして足すなら `extend` |

---

## 章末問題

### 問 1（読解）

```python
xs = ["c0000", "c0001", "c0002", "c0003", "c0004"]
```

次の結果は？ `xs[1:4]` / `xs[:2]` / `xs[-2:]` / `xs[3:100]` / `len(xs[5:])`

### 問 2（読解）

```python
order = {"顧客名": "ダミー商事", "品名": "", "数量": 0}
detail = {"品名": "VersaPro", "数量": 2}
merged = {**detail, **order}
```

`merged["品名"]` と `merged["数量"]` の値は？ dummy_csv の書き方（`{**order, **detail}`）と比べて何がまずいですか？

### 問 3（読解）

`textmatch.py` の `bigrams()` に `"ささら"` を渡したときの戻り値は？ `"あ"` と `""` のときは？

### 問 4（書く）

支払方法のリスト `methods = ["振込", "振込", "代引き", "振込", "クレジット"]` から、
**方法ごとの件数を数える辞書** を `dict.get` を使って作ってください。
結果: `{'振込': 3, '代引き': 1, 'クレジット': 1}`

### 問 5（書く）

`CLOSING = [("当月末", "翌月末", "001"), ("20日", "翌月20日", "002")]` を `for` で回し、
締日と ID だけを `001: 当月末` の形で表示してください（支払日は `_` で捨てる）。

### 問 6（考える）

次の関数を呼ぶと何が起きますか？ どう直しますか？

```python
def find_rows(conn, model):
    return conn.execute("SELECT * FROM chunks WHERE model = %s", (model)).fetchall()
```

---

## 回答

<details>
<summary>問 1 の回答</summary>

| 式 | 結果 |
|---|---|
| `xs[1:4]` | `['c0001', 'c0002', 'c0003']`（4 番目は含まない） |
| `xs[:2]` | `['c0000', 'c0001']` |
| `xs[-2:]` | `['c0003', 'c0004']`（後ろから 2 個） |
| `xs[3:100]` | `['c0003', 'c0004']`（範囲外でもエラーにならない） |
| `len(xs[5:])` | `0` |

</details>

<details>
<summary>問 2 の回答</summary>

- `merged["品名"]` は `""`、`merged["数量"]` は `0`
- 後ろに置いた `order` が勝つので、**明細の値が案件側の空の値で上書きされて消えます**。
  「共通の土台（order）を先に、個別の値（detail）を後に」が正しい順番です。
  dummy_csv は `{**order, **detail}` なので明細の値が優先されます。

</details>

<details>
<summary>問 3 の回答</summary>

- `"ささら"` → `{'ささ': 1, 'さら': 1}`
- `"あ"` → `{'あ': 1}`（2 文字未満は、その文字自身を 1 回）
- `""` → `{}`（`{text: 1} if text else {}` の `else` 側）

</details>

<details>
<summary>問 4 の回答</summary>

```python
methods = ["振込", "振込", "代引き", "振込", "クレジット"]
counts: dict[str, int] = {}
for m in methods:
    counts[m] = counts.get(m, 0) + 1
print(counts)  # {'振込': 3, '代引き': 1, 'クレジット': 1}
```

textmatch の `bigrams()` とまったく同じ形です。実務では標準ライブラリの `collections.Counter(methods)` を
使うことも多いです（第 14 章）。

</details>

<details>
<summary>問 5 の回答</summary>

```python
CLOSING = [("当月末", "翌月末", "001"), ("20日", "翌月20日", "002")]
for closing, _, terms_id in CLOSING:
    print(f"{terms_id}: {closing}")
# 001: 当月末
# 002: 20日
```

`for` の変数の位置でもアンパックが使えます。

</details>

<details>
<summary>問 6 の回答</summary>

`(model)` はカンマが無いので **タプルではなくただの文字列** です。
psycopg（3 系）は文字列のパラメータを受け付けず `TypeError` になります。
psycopg2 のような古いライブラリでは、文字列を「1 文字ずつの値の並び」として扱おうとし、
プレースホルダの数（1 個）と値の数（文字数）が合わないという別のエラーになります。

```python
return conn.execute("SELECT * FROM chunks WHERE model = %s", (model,)).fetchall()
# または [model] とリストで渡す
```

</details>
