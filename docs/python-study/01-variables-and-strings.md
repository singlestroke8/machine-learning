# 第 1 章 変数・データ型・文字列（f 文字列）

頻度: ★★★★★ — どのファイルにも、ほぼ全行に出てくる土台です。

この章で読めるようになるコード:

```python
CHUNK_MAX = 450
c["id"] = f"c{i:04d}"
print(f"text length: min={min(lens)} max={max(lens)} avg={sum(lens) / len(lens):.0f}")
"heading": heading or "",
per_file = -(-len(all_rows) // len(targets))  # 切り上げ
```

---

## 1.1 変数と「定数」

### 題材コード

`scripts/build_kb.py:18-19`

```python
CHUNK_MAX = 450  # 見出しを除く本文の最大文字数。長いと意味が薄まり、検索に引っかかりにくくなる
CHUNK_MIN = 60   # これ未満はボタン文言などのノイズとして捨てる
```

### 解説

- Python の変数は **宣言キーワードなし** で、代入した瞬間に生まれます（`let` / `const` / `$` は不要）。
- `#` から行末まではコメント。
- **Python には定数を作る構文がありません。** 「全部大文字の名前は定数として扱い、書き換えない」という
  **命名の約束（慣習）** で表現します。`CHUNK_MAX = 999` と書き換えてもエラーにはなりません。
- 名前の付け方は `snake_case`（小文字 + アンダースコア）が標準です（PEP 8 という公式スタイルガイド）。
  JS の `camelCase` で書くと、レビューで指摘されます。

| 用途 | Python の慣習 | 例 |
|---|---|---|
| 変数・関数 | `snake_case` | `cache_hits`, `split_long()` |
| 定数 | `UPPER_SNAKE` | `CHUNK_MAX`, `DEFAULT_MODEL` |
| クラス | `PascalCase` | `Embedder`, `MockEmbedder` |
| 内部用（外から使わないで） | 先頭に `_` | `_STRIP`, `_cache_path()` |

> **JS / PHP との比較**
> JS の `const` や PHP の `define()` / `const` に当たる「書き換え禁止」の仕組みは Python にはありません。
> 守るのは人間（とリンター）です。

---

## 1.2 基本のデータ型

| 型 | 例 | JS | PHP |
|---|---|---|---|
| `int` 整数 | `450`, `-1`, `10**13` | `number`（整数も小数も同じ） | `int` |
| `float` 小数 | `0.85`, `1e-3` | `number` | `float` |
| `str` 文字列 | `"query"`, `'document'` | `string` | `string` |
| `bool` 真偽値 | `True`, `False`（**先頭大文字**） | `true` / `false` | `true` / `false` |
| `None` 値なし | `None` | `null` / `undefined` | `null` |

- `int` には **桁数の上限がありません**（`10**100` もそのまま計算できる）。dummy_csv の
  `rng.randint(10**12, 10**13 - 1)`（13 桁の JAN コード）も、JS のような精度落ちを気にせず書けます。
- 型を調べるには `type(x)`。REPL で `type(0.85)` と打つと `<class 'float'>` と出ます。
- **自動の型変換はほとんど起きません。**

```python
>>> "1" == 1          # PHP の == なら true。Python は型が違えば等しくない
False
>>> "件数: " + 3       # JS なら "件数: 3"。Python はエラー
TypeError: can only concatenate str (not "int") to str
>>> "件数: " + str(3)  # 明示的に変換する
'件数: 3'
>>> 1 == 1.0          # 数値どうしは比較できる
True
```

PHP の `==` の緩い比較や、JS の `+` による暗黙の文字列化に慣れていると最初は戸惑いますが、
「勝手に変換しないので、変なバグが起きにくい」と考えると Python の方針が分かりやすくなります。

---

## 1.3 数値の演算

### 題材コード

`PivotAggregationSystem/scripts/generate_dummy_csv.py:530`

```python
per_file = -(-len(all_rows) // len(targets))  # 切り上げ
```

`reference/okubo-shamisen/textmatch.py:39`

```python
na = sum(v * v for v in ga.values()) ** 0.5
```

### 解説

| 演算子 | 意味 | 例 | 結果 |
|---|---|---|---|
| `/` | 割り算（**常に float**） | `6 / 2` | `3.0` |
| `//` | 切り捨て除算（床関数） | `7 // 2` / `-7 // 2` | `3` / `-4` |
| `%` | 余り | `seq % 1000` | |
| `**` | べき乗 | `2 ** 10` / `16 ** 0.5` | `1024` / `4.0`（平方根） |

- `/` は割り切れても `3.0`（float）になります。整数の結果がほしいときは `//`。
- `//` は「小さい方へ丸める」ので、負の数では `-7 // 2 == -4` です（0 方向ではない）。
- **`-(-a // b)` は切り上げ除算の定番イディオム**です。
  `-a // b` で「負の方向へ切り捨て」→ 符号を戻すと「正の方向へ切り上げ」になります。
  `math.ceil(a / b)` でも書けますが、float を経由しないので巨大な整数でも誤差が出ません。
- `** 0.5` は平方根。textmatch では「ベクトルの長さ（ノルム）」を求めています（第 5 章で詳しく）。

### round() の注意

`scripts/export_kb.py:57`

```python
"vec": [round(x, ROUND) for x in json.loads(emb)],
```

`round(x, 4)` は小数 4 桁に丸めます。ただし `round()` は **「偶数への丸め（銀行丸め）」** です。

```python
>>> round(2.5), round(3.5)
(2, 4)                 # 2.5 → 2 になる！（四捨五入ではない）
>>> round(1234.5, -2)  # 桁数にマイナスを渡すと 10 の位・100 の位で丸める
1200.0                 # 桁数を渡すと float が返る
```

dummy_csv で販売単価を作る行が `int(...)` で包まれているのは、この「float が返る」対策です。

`generate_dummy_csv.py:269`

```python
price = int(round(cost * rng.uniform(lo, hi), -2))
```

お金の計算など **正確な四捨五入が必要な場面では `round()` を使わず `Decimal` を使います**
（dummy_csv の `yen_round()`。第 14 章）。

---

## 1.4 文字列と f 文字列

### 題材コード

`scripts/build_kb.py:96-97, 106`

```python
for i, c in enumerate(chunks):
    c["id"] = f"c{i:04d}"
...
print(f"text length: min={min(lens)} max={max(lens)} avg={sum(lens) / len(lens):.0f}")
```

### 解説

- 文字列は `"..."` でも `'...'` でも同じ意味です（PHP のようにダブルクォートだけ変数展開する、という区別はない）。
- 変数を埋め込むには、先頭に `f` を付けた **f 文字列** を使います。`{ }` の中には **任意の式** が書けます。
  - JS のテンプレートリテラル `` `c${i}` `` / PHP の `"c{$i}"` に相当します。
- `{式:書式}` のようにコロンの後ろで **書式** を指定できます。実務で使うのはほぼ次の表だけです。

| 書式 | 意味 | 例 | 結果 |
|---|---|---|---|
| `:04d` | 整数を 4 桁ゼロ埋め | `f"c{7:04d}"` | `c0007` |
| `:.0f` | 小数 0 桁（丸め） | `f"{93.6:.0f}"` | `94` |
| `:.2f` | 小数 2 桁 | `f"{0.12345:.2f}"` | `0.12` |
| `:.1%` | 100 倍して % 表示 | `f"{0.8567:.1%}"` | `85.7%` |
| `:,` | 3 桁カンマ | `f"{1234567:,}"` | `1,234,567` |
| `:>5` / `:<4` | 右寄せ / 左寄せ | `f"{3:>5}"` | `    3` |
| `:%Y%m%d` | 日付の書式（date / datetime） | `f"{booked:%Y%m%d}"` | `20251002` |

`c{i:04d}` は「`c` + 4 桁ゼロ埋めの通し番号」で、`c0000, c0001, ...` という ID を作っています。
桁を揃えると、文字列として並べ替えても数値順になるという利点があります。

dummy_csv では、日付の書式指定と組み合わせて ID を作っています。

`generate_dummy_csv.py:419`

```python
"指示書No.": f"DUMMY{booked:%Y%m%d}{seq % 1000:03d}" + rng.choices(...)[0],
```

`booked` は日付（`date`）なので `:%Y%m%d` で `20251002` のようになり、`seq % 1000:03d` で 000〜999 の 3 桁が続きます。

### f 文字列の中のクォート

`reference/okubo-shamisen/eval.py:203`

```python
out.append(f"| {r['tau']:.2f} | {r['coverage']*100:.0f}% | {r['false_answers']}/{n_neg} |")
```

外側を `"`、`{}` の中の辞書のキーを `'` にしています。Python 3.11 までは **外側と同じクォートを
`{}` の中で使えなかった** ため、こう書くのが習慣です（3.12 からは同じクォートも可）。
このリポジトリの Python ツール用コンテナ（`docker/python/Dockerfile`）は **Python 3.11** なので、ここでは外と中でクォートを変えるのが必須です。

### 複数行文字列と docstring

`"""` で囲むと改行を含む文字列が書けます。ファイルや関数の先頭に置いた `"""..."""` は
**docstring（説明文）** と呼ばれ、`help(関数)` で表示されます。

`scripts/build_kb.py:32-36`

```python
def split_long(text: str) -> list[str]:
    """CHUNK_MAX を超える本文を、段落（改行）の境目で分ける。

    段落の途中では切らないので、1 段落だけで CHUNK_MAX を超える場合はそのまま 1 チャンクになる。
    """
```

1 行目に要約、空行を挟んで詳細、が標準の書き方です。PHP の PHPDoc（`/** ... */`）の役割に近いですが、
Python では **文字列そのもの** なので実行時に `split_long.__doc__` で取り出せます。

### 長い文字列を分けて書く

`scripts/export_kb.py:39-43`

```python
rows = conn.execute(
    "SELECT id, page_key, title, heading, text, embedding::text"
    " FROM chunks WHERE model = %s ORDER BY id",
    (args.model,),
).fetchall()
```

**文字列リテラルを並べて書くと、自動で 1 つにつながります**（`+` 不要）。
2 行目の先頭に半角スペース `" FROM"` があるのは、つなげたときに `embedding::textFROM` とくっつかないためです。
長い SQL やメッセージを読みやすく折り返す、実務の定番テクニックです。

---

## 1.5 よく使う文字列メソッド

題材に出てくるものを、出現頻度順に並べます。

| メソッド | 何をするか | 題材での使用例 |
|---|---|---|
| `s.strip()` | 前後の空白・改行を除く | `p.strip()`（build_kb） |
| `s.split(sep)` | 区切り文字で分けてリストに | `text.split("\n")`（build_kb） |
| `sep.join(list)` | リストを区切り文字でつなぐ | `"\n".join(buf)`（build_kb） |
| `s.splitlines()` | 行ごとに分ける（末尾の改行を気にしなくてよい） | `body.splitlines()`（build_kb） |
| `s.replace(a, b)` | 置換 | `.replace("{{SCHOOL}}", school_name)`（eval） |
| `s.lstrip(chars)` | 左端から指定文字を除く | `line.lstrip("#")`（build_kb） |
| `s.lower()` | 小文字に | textmatch の `normalize` |
| `s.encode("utf-8")` | バイト列に変換 | embedders のハッシュ計算 |
| `len(s)` | 文字数（関数であってメソッドではない） | `len(cur)`（build_kb） |

### join の向きに注意

```python
"\n".join(buf)       # Python:  区切り文字.join(リスト)
```

```javascript
buf.join("\n")       // JS:      配列.join(区切り文字)
```

```php
implode("\n", $buf); // PHP
```

Python だけ **区切り文字の方にメソッドがある** ので、最初は必ず書き間違えます。
「区切り文字がリストをつなぐ」と覚えてください。

### split と splitlines の違い

```python
>>> "a\nb\n".split("\n")
['a', 'b', '']        # 末尾の改行の後ろに空文字が残る
>>> "a\nb\n".splitlines()
['a', 'b']            # 残らない。\r\n（Windows の改行）も扱える
```

build_kb はファイル本文を行に分けるとき `splitlines()`、段落の連結結果を分けるとき `split("\n")` と
使い分けています。ファイルを行に分けるなら `splitlines()` が安全です。

### lstrip は「文字の集合」を除く

`scripts/build_kb.py:79`

```python
heading = line.lstrip("#").strip()
```

`"## 料金について"` → `lstrip("#")` で `" 料金について"` → `strip()` で `"料金について"`。
`lstrip("#")` は「`#` という **文字** を左端から何個でも」除きます。「`#` という文字列を 1 回」ではありません。
`lstrip("ab")` は `"ab"` の並びではなく「`a` か `b` の文字」を除くので注意してください
（接頭辞を 1 回だけ除きたいなら `s.removeprefix("ab")`、Python 3.9+）。

---

## 1.6 真偽値の判定（truthy / falsy）と `or`

### 題材コード

`scripts/build_kb.py:41, 45-46`

```python
if cur and len(cur) + len(para) > CHUNK_MAX:
...
cur = f"{cur}\n{para}" if cur else para
if cur:
    parts.append(cur)
```

`scripts/export_kb.py:55` / `scripts/load_kb.py:54`

```python
"heading": heading or "",          # export_kb: None なら "" にする
c["heading"] or None               # load_kb:   "" なら None にする（DB に NULL で入れる）
```

### 解説

`if` の条件に置いたとき **偽として扱われる値（falsy）** は次のとおりです。それ以外はすべて真です。

- `False`, `None`
- 数値の `0`, `0.0`
- 空の入れ物: `""`, `[]`, `{}`, `()`, `set()`

`if cur:` は「`cur` が空文字でなければ」という意味です。`if cur != "":` と書くより Python らしい書き方です。

`and` / `or` は JS の `&&` / `||` と同じく **真偽値ではなく、決め手になった方の値そのもの** を返します。

```python
>>> "" or None        # 左が falsy → 右を返す
>>> "料金" or None
'料金'
>>> 0 or 1.0
1.0
```

これを利用したのが `heading or ""`（None を空文字に寄せる）と `c["heading"] or None`（空文字を None に寄せる）です。
同じ `or` で逆向きの変換をしている点に注目してください。
export_kb は JSON（Lambda 側で文字列として扱いたい）、load_kb は DB（「見出しなし」を NULL で表したい）と、
**行き先の都合に合わせて** 形を変えています。

> **JS / PHP との比較**
> - JS の `??`（null のときだけ右を使う）に当たる演算子は **Python にはありません**。
>   `or` は `0` や `""` でも右に切り替わる（JS の `||` と同じ）ので、`0` を有効な値として扱いたいときは
>   `x if x is not None else 既定値` と書きます。
> - PHP の `?:`（エルビス演算子）が Python の `or` に近い動きです。
> - 空配列 `[]` は JS では **truthy** ですが、Python では **falsy** です。`if rows:` で「1 件以上あれば」が書けます。

### `None` との比較は `is`

```python
if reply is None:          # eval.py:117
    return None
```

`None` かどうかは `==` ではなく **`is None` / `is not None`** で比べるのが決まりです。
`is` は「同じオブジェクトか」を比べる演算子で、`None` はプログラム中に 1 つしか存在しないためこう書きます。

---

## 1.7 落とし穴まとめ

| 落とし穴 | 例 | 対策 |
|---|---|---|
| `/` は必ず float | `len(x) / 2` がリストの添字に使えない | 整数がほしいときは `//` |
| `round()` は四捨五入ではない | `round(2.5) == 2` | お金は `Decimal`（第 14 章） |
| 文字列と数値を `+` でつなぐ | `"件数" + 3` → TypeError | f 文字列で `f"件数{n}"` |
| `true` と書く | `NameError: name 'true' is not defined` | `True` / `False` / `None` は先頭大文字 |
| `or` で 0 が消える | `count or 10` は `count=0` でも 10 | `count if count is not None else 10` |
| `lstrip` に文字列を渡す | `"abba".lstrip("ab") == ""` | 接頭辞なら `removeprefix` |

---

## 章末問題

### 問 1（読解）次の式の結果を答えてください

```python
f"c{42:04d}"
7 / 2
7 // 2
-(-7 // 2)
-(-8 // 2)
f"{0.4567:.1%}"
```

### 問 2（読解）次の 3 つの結果は？

```python
heading = ""
a = heading or None
heading = "教室の場所"
b = heading or None
count = 0
c = count or 10
```

### 問 3（読解）`"### アクセス・料金 ".lstrip("#").strip()` の結果は？
また `"#1 の使い方".lstrip("#1")` の結果は？

### 問 4（書く）

変数 `hits = 7`、`total = 33` があります。`ヒット率: 21.2%（7/33）` と表示する 1 行を f 文字列で書いてください。

### 問 5（書く）

`build_kb.py:106` を真似て、リスト `lens = [120, 450, 88, 301]` から
`min=88 max=450 avg=240` と表示してください（平均は小数 0 桁）。

### 問 6（考える）

`load_kb.py` は `c["heading"] or None` で空文字を None に変えて DB に入れ、
`export_kb.py` は `heading or ""` で None を空文字に戻しています。
最初から空文字のまま DB に入れておけば両方の変換が要らないのに、なぜこうしているのでしょうか。
あなたの考えを書いてください（正解は 1 つではありません）。

---

## 回答

<details>
<summary>問 1 の回答</summary>

| 式 | 結果 | 理由 |
|---|---|---|
| `f"c{42:04d}"` | `'c0042'` | 4 桁ゼロ埋め |
| `7 / 2` | `3.5` | `/` は常に float |
| `7 // 2` | `3` | 切り捨て |
| `-(-7 // 2)` | `4` | `-7 // 2 == -4` → 符号反転で 4（切り上げ） |
| `-(-8 // 2)` | `4` | 割り切れるときはそのまま |
| `f"{0.4567:.1%}"` | `'45.7%'` | 100 倍して小数 1 桁 |

</details>

<details>
<summary>問 2 の回答</summary>

- `a` は `None`（`""` は falsy なので右側）
- `b` は `'教室の場所'`
- `c` は `10` — `0` も falsy なので右に切り替わります。「0 件」を正しく扱いたいなら
  `c = count if count is not None else 10` と書きます。

</details>

<details>
<summary>問 3 の回答</summary>

- `'アクセス・料金'` — 左の `#` を全部除いた後、`strip()` で前後の空白を除く
- `' の使い方'` — `lstrip("#1")` は「`#` か `1` の文字」を左から除くので、`#1` の後ろの空白は残る。
  `"1"` が 2 文字目にあるから除かれたのではなく、「`#`」「`1`」どちらも除く対象の文字だから除かれています。
  `"#11#1 の使い方"` でも結果は同じ `' の使い方'` になります。

</details>

<details>
<summary>問 4 の回答</summary>

```python
hits, total = 7, 33
print(f"ヒット率: {hits / total:.1%}（{hits}/{total}）")
# ヒット率: 21.2%（7/33）
```

`:.1%` が 100 倍してくれるので、自分で `* 100` しなくて済みます。
eval.py は `{r['coverage']*100:.0f}%` と自分で 100 倍していますが、どちらでも構いません。

</details>

<details>
<summary>問 5 の回答</summary>

```python
lens = [120, 450, 88, 301]
print(f"min={min(lens)} max={max(lens)} avg={sum(lens) / len(lens):.0f}")
# min=88 max=450 avg=240
```

`sum(lens) / len(lens)` は `239.75`。`:.0f` で丸めて `240` になります。

</details>

<details>
<summary>問 6 の回答例</summary>

- **DB では「値がない」を NULL で表すのが素直** だからです。`heading IS NULL` で「見出しのないチャンク」を
  検索でき、空文字と NULL が混在する状態（どちらで検索すべきか迷う）を避けられます。
- 一方、JSON を読む TypeScript / Lambda 側では `heading` が常に `string` の方が型が単純になり、
  `null` チェックを書かずに済みます。
- つまり **保存の形（DB）と受け渡しの形（JSON）を、それぞれの使い手に合わせて変えている** ということです。
  境界で形を整え、内部ではいつも同じ形を前提にする、という考え方は実務で頻繁に出てきます。

</details>
