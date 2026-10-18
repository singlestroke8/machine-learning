# 第 3 章 制御構文（if / for / enumerate / zip / 条件式）

頻度: ★★★★★

この章で読めるようになるコード:

```python
for i, t in enumerate(texts):
for (q, expect), v in zip(positives, pos_vecs_cfg):
cur = f"{cur}\n{para}" if cur else para
for attempt in range(4):
```

---

## 3.1 インデントがブロック

Python には `{ }` がありません。**`:` で始まり、インデント（字下げ）が続く範囲がブロック** です。

```python
if p.exists():                      # 行末の : を忘れると SyntaxError
    out[i] = json.loads(p.read_text())
    self.cache_hits += 1            # ここまでが if の中
else:
    misses.append(i)
```

- インデントは **スペース 4 個** が標準（タブと混ぜない）。
- 中身が空のブロックは書けないので、何もしないときは `pass` と書きます。
- `i++` はありません。`i += 1` と書きます。

> **JS / PHP との比較**
> `{ }` の代わりにインデント、`else if` の代わりに `elif`、`&&` `||` `!` の代わりに `and` `or` `not`。
> 条件を `( )` で囲む必要もありません（囲んでもエラーではないが、書かないのが普通）。

---

## 3.2 if / elif / else と比較演算子

### 題材コード

`reference/okubo-shamisen/eval.py:302-311`

```python
for (q, expect), v in zip(positives, pos_vecs_cfg):
    tier, got, cands = decide(q, v, faq, cfg, router)
    if tier == "pending":
        pos_pending += expect in cands
        continue
    if got == expect:
        pos_ok += 1
    elif got:
        pos_wrong += 1
        pos_lines.append(f"- [誤答] {q} → 期待 {expect} / {tier} で {got}")
```

### 解説

評価用の質問を 1 問ずつ判定し、「正解」「誤答」「判定段待ち」を数えています。

- `==` は値の比較。Python は **型の変換をしないので、PHP の `===` に近い** と考えて構いません。
- `elif got:` — 「正解ではなかった（`got != expect`）が、何かしらの FAQ を返した（`got` が空でない）」= 誤答。
  `got` が `None` や `""` の場合（何も答えなかった）はどちらにも当てはまらず、何も数えません。
- `continue` はループの残りを飛ばして次の周回へ（JS / PHP と同じ）。

### `in` / `not in`

```python
if input_type not in ("document", "query"):        # embedders.py:46
    raise ValueError(...)
if e.code in (429, 500, 502, 503) and attempt < 3:  # ref embedders.py:121
```

`x in (a, b, c)` は「x が a, b, c のどれかと等しいか」。`x == a or x == b or x == c` を短く書けます。
文字列にも使えて、`"在庫生産" in customer` は「部分文字列として含むか」です（PHP の `str_contains`、JS の `includes`）。

### `True` も数として足せる

`pos_pending += expect in cands` は少し変わった書き方です。

- `expect in cands` は `True` か `False`。
- Python の `bool` は `int` の仲間で、**`True` は 1、`False` は 0** として足し算できます。
- つまり「`expect` が候補に入っていたら 1 足す」を 1 行で書いています。

```python
>>> True + True + False
2
```

読みやすさを優先するなら `if expect in cands: pos_pending += 1` と書いても構いません。
eval.py には同じ形の `neg_to_router += bool(cands)`（候補が 1 件以上あれば 1 足す）もあります。

### 比較の連結

```python
if 0.6 <= score < 0.9:      # 「0.6 以上 0.9 未満」を数学と同じ形で書ける
```

JS / PHP では `0.6 <= score && score < 0.9` と書くところです。

---

## 3.3 for 文と range

### 題材コード

`scripts/embedders.py:61-62`

```python
for start in range(0, len(misses), BATCH):
    idx = misses[start : start + BATCH]
```

### 解説

Python の `for` は **「入れ物から 1 個ずつ取り出す」** 形しかありません（JS の `for...of`、PHP の `foreach`）。
`for (let i = 0; i < n; i++)` のような C 言語風の for はなく、回数で回したいときは `range()` を使います。

| 書き方 | 生成される値 |
|---|---|
| `range(5)` | 0, 1, 2, 3, 4 |
| `range(2, 5)` | 2, 3, 4 |
| `range(0, 300, 128)` | 0, 128, 256 |
| `range(5, 0, -1)` | 5, 4, 3, 2, 1 |

`range(0, len(misses), BATCH)` は「0 から 128 刻み」。スライスと組み合わせて **128 件ずつのバッチ処理** になります。

ループ変数を使わないときは `_` にするのが慣習です。

`generate_dummy_csv.py:257`

```python
for _ in range(rng.choices(PRODUCT_LINE_COUNTS, weights=PRODUCT_LINE_WEIGHTS)[0]):
```

「明細を 1〜6 行のどれかの回数だけ作る」という意味で、何周目かは使っていません。

### 文字列も、辞書も、ファイルも for で回せる

```python
for line in body.splitlines():     # 行のリスト
for path in sorted(PAGES_DIR.glob("*.md")):   # ファイルの一覧
for k, v in ga.items():            # 辞書のキーと値
for ch in "三味線":                # 文字列は 1 文字ずつ
```

`for` で回せるものを **イテラブル（iterable）** と呼びます。Python では非常に多くのものがイテラブルで、
「とりあえず for に入れてみる」が通用します。

---

## 3.4 enumerate — 番号付きで回す

### 題材コード

`scripts/build_kb.py:96-97`

```python
for i, c in enumerate(chunks):
    c["id"] = f"c{i:04d}"
```

### 解説

`enumerate(xs)` は `(0, xs[0]), (1, xs[1]), ...` という **(番号, 要素) の組** を順に返します。
それを `for i, c in ...` でアンパックしています。

```python
>>> list(enumerate(["a", "b"]))
[(0, 'a'), (1, 'b')]
>>> list(enumerate(["a", "b"], start=1))   # 1 始まりにもできる
[(1, 'a'), (2, 'b')]
```

> **比較**
> PHP: `foreach ($chunks as $i => $c)` / JS: `for (const [i, c] of chunks.entries())`
>
> 初心者は `for i in range(len(chunks)): c = chunks[i]` と書きがちですが、
> **Python では enumerate を使うのが正解** です。レビューで必ず指摘されます。

ループ内で `c["id"] = ...` と書き換えると、**元のリストの中の辞書が変わります**。
`c` は辞書のコピーではなく、同じ辞書を指しているからです（第 2 章の「参照」）。

---

## 3.5 zip — 2 つのリストを並べて回す

### 題材コード

`scripts/embedders.py:72`

```python
for i, vec in zip(idx, res.embeddings):
```

`scripts/load_kb.py:53-56`

```python
cur.executemany(INSERT_SQL, [
    (c["page_key"], c["title"], c["heading"] or None, c["text"], emb.model, np.array(v))
    for c, v in zip(chunks, vecs)
])
```

### 解説

`zip(a, b)` は `(a[0], b[0]), (a[1], b[1]), ...` と、**同じ位置どうしを組にして** 返します。

- embedders: 「API に送った位置 `idx`」と「返ってきたベクトル」を組にして、元の位置に戻す
- load_kb: 「チャンク」と「そのベクトル」を組にして、DB の 1 行分のタプルを作る

どちらも **2 つのリストの順番が揃っている** ことが前提です。
embedders.py が `[None] * len(texts)` で位置を守っていたのは、ここで zip するためでもあります。

### 入れ子のアンパック

`reference/okubo-shamisen/eval.py:302`

```python
for (q, expect), v in zip(positives, pos_vecs_cfg):
```

`positives` の要素は `(質問, 期待するid)` のタプルです。zip すると `((質問, 期待id), ベクトル)` という
**入れ子の組** になるので、左辺も `(q, expect), v` と入れ子で受けています。

### zip の注意: 短い方で止まる

```python
>>> list(zip([1, 2, 3], ["a", "b"]))
[(1, 'a'), (2, 'b')]          # 3 は黙って捨てられる
>>> list(zip([1, 2, 3], ["a", "b"], strict=True))   # Python 3.10+
ValueError: zip() argument 2 is shorter than argument 1
```

長さが違うのはたいていバグなので、**長さが揃っているはずの zip には `strict=True` を付ける** と安全です
（題材のコードには付いていません。付けるとより堅くなる改善点です）。

---

## 3.6 条件式（三項演算子）

### 題材コード

`scripts/build_kb.py:45`

```python
cur = f"{cur}\n{para}" if cur else para
```

### 解説

Python の三項演算子は **`真のときの値 if 条件 else 偽のときの値`** の順です。

```python
cur = f"{cur}\n{para}" if cur else para     # Python
```

```javascript
cur = cur ? `${cur}\n${para}` : para;       // JS
```

```php
$cur = $cur ? "{$cur}\n{$para}" : $para;    // PHP
```

**条件が真ん中に来る** ので、英語の文のように「X if 条件 else Y（条件なら X、さもなくば Y）」と読みます。
ここでは「`cur` が空でなければ改行でつなぎ、空なら `para` をそのまま使う」。
先頭に余計な改行が付かないようにする工夫です。

### 条件式の連結

`generate_dummy_csv.py:327-333`

```python
shipping_status = (
    SHIPPING_SHIPPED
    if shipped
    else SHIPPING_PRODUCED
    if status_group == GROUP_PRODUCTION and rng.random() < 0.5
    else ""
)
```

`A if 条件1 else (B if 条件2 else C)` という連結で、if / elif / else と同じ意味になります。

```python
if shipped:
    shipping_status = SHIPPING_SHIPPED
elif status_group == GROUP_PRODUCTION and rng.random() < 0.5:
    shipping_status = SHIPPING_PRODUCED
else:
    shipping_status = ""
```

全体を `( )` で囲むと、中では自由に改行できます。3 段以上になるなら if 文で書いた方が読みやすい、
というのが一般的な感覚です。

---

## 3.7 break / continue — リトライのループを読む

### 題材コード

`reference/okubo-shamisen/embedders.py:115-124`

```python
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
```

### 解説

API 呼び出しを **最大 4 回** 試す、実務で頻出のリトライ処理です。

| attempt | 成功したら | 429 / 5xx だったら | それ以外のエラー |
|---|---|---|---|
| 0 | `break` でループを抜ける | 2 秒待って `continue`（次の周回へ） | `raise`（エラーをそのまま投げ直す） |
| 1 | 同上 | 4 秒待って次へ | 同上 |
| 2 | 同上 | 8 秒待って次へ | 同上 |
| 3 | 同上 | `attempt < 3` が偽 → `raise` | 同上 |

- 待ち時間が 2, 4, 8 秒と倍々に伸びるのは **指数バックオフ** と呼ばれる手法です。混んでいるサーバに連打しないため。
- `try` / `except` / `raise` は第 8 章、`with` は第 12 章で扱います。ここでは `break` と `continue` の流れを追えれば十分です。
- 再実装版の `scripts/embedders.py` では、このリトライを Voyage の公式 SDK に任せています
  （`voyageai.Client(max_retries=3)`）。**ライブラリにある機能は自分で書かない** のも実務の判断です。

### for ... else（参考）

Python には **`break` されずに最後まで回ったときだけ実行される** `else` 節があります。

```python
for attempt in range(4):
    if try_call():
        break
else:
    raise RuntimeError("4 回とも失敗")   # 1 度も break しなかったとき
```

知らないと読めないので紹介しましたが、読み手を迷わせやすいので多用はされません。

---

## 3.8 while と match（参考）

- `while 条件:` もあります。ただ、ほとんどの繰り返しは `for` で書けるため、
  題材のコードにも一度も出てきません。「終わりが回数で決まらない」ときだけ使います。
- Python 3.10 から `match` 文（パターンマッチ）が入りました。PHP 8 の `match` とは別物で、
  データの形で分岐できる強力な構文ですが、実務での使用頻度はまだ低めです。

---

## 3.9 アルゴリズムを追う — split_long

ここまでの知識で `scripts/build_kb.py` の `split_long()` が読めるはずです。

```python
def split_long(text: str) -> list[str]:
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
```

仮に `CHUNK_MAX = 10`、`CHUNK_MIN = 3` として、`"月謝は\n月4回で\n8000円です\nほか\n入会金"` を渡したときの動きです。

| para | `cur` の長さ + `para` の長さ | 動き | 処理後の `cur` |
|---|---|---|---|
| `月謝は` | （cur が空） | else: そのまま入れる | `月謝は` |
| `月4回で` | 3 + 4 = 7 | else: 改行でつなぐ | `月謝は\n月4回で`（8 文字） |
| `8000円です` | 8 + 7 = 15 > 10 | **確定** して新しく始める | `8000円です` |
| `ほか` | 7 + 2 = 9 | else: つなぐ | `8000円です\nほか`（10 文字） |
| `入会金` | 10 + 3 = 13 > 10 | **確定** | `入会金` |
| （ループ終了） | | 残りを確定 | |

結果: `['月謝は\n月4回で', '8000円です\nほか', '入会金']`

「溜めて、あふれそうになったら確定」という **バッファリング** は、ログの書き出し・バッチ送信・CSV の分割など、
実務のあちこちに出てくる形です。

---

## 章末問題

### 問 1（読解）次のループの出力は？

```python
for i, name in enumerate(["本社", "大阪支店", "福岡営業所"], start=1):
    if "支店" in name:
        continue
    print(i, name)
```

### 問 2（書き換え）次の JS を Python の 1 行に書き換えてください

```javascript
const label = score >= 0.9 ? "t1" : score >= 0.6 ? "router" : "none";
```

### 問 3（読解）

`split_long()` で `CHUNK_MAX = 10`、`CHUNK_MIN = 3` のとき、`"教室\nアクセス案内です\nあ"` の結果は？
（`len("教室") = 2`、`len("アクセス案内です") = 8`、`len("あ") = 1`）

### 問 4（書く）

`questions = ["月謝は？", "体験は？", "駐車場は？"]` と `answers = ["8000円", "あり", "なし"]` を zip で回して
`Q: 月謝は？ / A: 8000円` の形で表示してください。長さが違ったらエラーになるようにしてください。

### 問 5（書く）

`scores = [0.95, 0.42, 0.88, 0.61, 0.30]` のうち **0.6 以上のものの個数** を、for と if で数えてください。
次に、`True` が 1 として足せることを使って、`if` を使わずに書いてください。

### 問 6（考える）

リトライのループ（ref embedders.py:115-124）で、`attempt < 3` の条件を外すと何が起きますか？

---

## 回答

<details>
<summary>問 1 の回答</summary>

```
1 本社
3 福岡営業所
```

`enumerate` の番号は `continue` で飛ばしても進むので、大阪支店の 2 は欠番になります。

</details>

<details>
<summary>問 2 の回答</summary>

```python
label = "t1" if score >= 0.9 else "router" if score >= 0.6 else "none"
```

</details>

<details>
<summary>問 3 の回答</summary>

| para | 判定 | cur |
|---|---|---|
| `教室` | cur が空 → 入れる | `教室`（2） |
| `アクセス案内です` | 2 + 8 = 10、`> 10` ではない → つなぐ | `教室\nアクセス案内です`（11） |
| `あ` | 11 + 1 = 12 > 10 → 確定 | `あ` |
| 終了 | 残りを確定 | |

`parts = ['教室\nアクセス案内です', 'あ']`。最後に `len >= 3` で絞るので `'あ'` は捨てられ、
結果は `['教室\nアクセス案内です']`。

`len(cur)` には改行 1 文字も含まれるので、「つないだ結果は CHUNK_MAX をわずかに超えうる」ことにも
気付けたら完璧です（`len(cur) + len(para)` は、つなぐときに入る改行の 1 文字を数えていない）。

</details>

<details>
<summary>問 4 の回答</summary>

```python
questions = ["月謝は？", "体験は？", "駐車場は？"]
answers = ["8000円", "あり", "なし"]
for q, a in zip(questions, answers, strict=True):
    print(f"Q: {q} / A: {a}")
```

</details>

<details>
<summary>問 5 の回答</summary>

```python
scores = [0.95, 0.42, 0.88, 0.61, 0.30]

n = 0
for s in scores:
    if s >= 0.6:
        n += 1
print(n)  # 3

n = 0
for s in scores:
    n += s >= 0.6   # True なら 1、False なら 0 が足される
print(n)  # 3
```

第 5 章では `sum(1 for s in scores if s >= 0.6)` という 1 行の書き方を学びます（eval.py の `sweep()` がこの形）。

</details>

<details>
<summary>問 6 の回答</summary>

4 回目（`attempt == 3`）も 429 / 5xx なら `continue` してループが終わり、`break` も `raise` もされずに
ループを抜けます。すると `data` が一度も代入されていない（または前のバッチの値が残っている）状態で
次の行 `sorted(data["data"], ...)` に進み、`NameError`、最悪の場合は **前のバッチの結果を黙って使う** バグになります。
`attempt < 3` は「最後の 1 回は失敗を握りつぶさず投げ直す」ための条件です。

</details>
