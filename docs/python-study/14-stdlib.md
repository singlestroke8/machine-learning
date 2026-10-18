# 第 14 章 標準ライブラリ（datetime / decimal / random / hashlib / itertools / collections）

頻度: ★★★☆☆ — 「Python は電池付き（batteries included）」と言われるとおり、標準ライブラリだけでかなりのことができます。
dummy_csv は **標準ライブラリだけ** で 544 行のツールを作っています（docstring に「追加インストール不要」とある）。

この章で読めるようになるコード:

```python
first_next = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
rng = random.Random(args.seed)
key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
writer.writerows(islice(rows_iter, per_file))
```

---

## 14.1 datetime — 日付と時刻

### 題材コード

`PivotAggregationSystem/scripts/generate_dummy_csv.py:186-204`

```python
def ymd(d: date | None) -> str:
    """日付を元システムの形式(2025/10/02)にする。"""
    return d.strftime("%Y/%m/%d") if d else ""


def ymdhms_slash(d: datetime | None) -> str:
    """日時を元システムの形式(2025/10/30 14:47:09)にする。"""
    return d.strftime("%Y/%m/%d %H:%M:%S") if d else ""

...

def month_end(d: date) -> date:
    """その月の末日。"""
    first_next = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return first_next - timedelta(days=1)
```

### 解説

| クラス | 表すもの | 例 |
|---|---|---|
| `date` | 日付だけ | `date(2025, 10, 2)` / `date.today()` |
| `datetime` | 日付 + 時刻 | `datetime(2025, 10, 30, 14, 47, 9)` / `datetime.now()` |
| `timedelta` | 期間（差） | `timedelta(days=32)` / `timedelta(seconds=3600)` |
| `timezone` | タイムゾーン | `timezone.utc` |

**日付の足し引きは `timedelta`** で行います。

```python
booked + timedelta(days=32)           # 32 日後
today - timedelta(days=540)           # 540 日前
d2 - d1                               # 日付どうしの差は timedelta になる（.days で日数）
```

**文字列との変換**

| 方向 | メソッド | 例 |
|---|---|---|
| 日付 → 文字列 | `d.strftime("%Y/%m/%d")`（**f**ormat） | `'2025/10/02'` |
| 日付 → 文字列 | `f"{d:%Y/%m/%d}"`（f 文字列でも同じ） | `'2025/10/02'` |
| 文字列 → 日時 | `datetime.strptime(s, "%Y/%m/%d %H:%M:%S")`（**p**arse） | |
| ISO 8601 形式 | `dt.isoformat()` / `datetime.fromisoformat(s)` | `'2026-10-18T13:59:08+00:00'` |

| 書式 | 意味 | 書式 | 意味 |
|---|---|---|---|
| `%Y` | 4 桁の年 | `%H` | 時（24 時間） |
| `%m` | 2 桁の月 | `%M` | 分 |
| `%d` | 2 桁の日 | `%S` | 秒 |

PHP の `date('Y/m/d')` と書式の文字が違うので注意してください（PHP は `%` が無く、`m` / `d` / `H` / `i` / `s`）。

### month_end — 月末日を求める定番の工夫

1. `d.replace(day=28)` — その月の 28 日にする（**どの月にも 28 日はある**）
2. `+ timedelta(days=4)` — 4 日足す。28 + 4 = 32 日目なので、**必ず翌月** に入る
3. `.replace(day=1)` — 翌月の 1 日
4. `- timedelta(days=1)` — その前日 = **今月の末日**

```python
>>> month_end(date(2024, 2, 10))
datetime.date(2024, 2, 29)      # うるう年も正しい
>>> month_end(date(2025, 12, 31))
datetime.date(2025, 12, 31)     # 年をまたいでも正しい
```

「月の日数」を表で持たずに済む、うまい書き方です。標準ライブラリの `calendar.monthrange(年, 月)[1]` でも末日の日を得られます。

`replace()` は **元の日付を変えず、一部を置き換えた新しい日付を返します**（日付は変更できない値）。

### datetime.combine — 日付に時刻を付ける

`generate_dummy_csv.py:321-323`

```python
registered = datetime.combine(today - timedelta(days=rng.randint(0, lookback_days(today))), datetime.min.time()) + timedelta(
    seconds=rng.randint(8 * 3600, 19 * 3600)
)
```

- `datetime.combine(日付, 時刻)` — `date` と `time` から `datetime` を作る
- `datetime.min.time()` — 00:00:00
- `+ timedelta(seconds=8 * 3600 〜 19 * 3600)` — 8 時〜19 時のどこか

→ 「過去 1 年半のどこかの日の、営業時間内のどこかの時刻」という、それらしい登録日時を作っています。

### タイムゾーン

`scripts/export_kb.py:68`

```python
"exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
```

`datetime.now()` は **タイムゾーン情報を持たない**（naive な）日時を返します。サーバの設定次第で日本時間にも UTC にもなります。
**ファイルやログに残す時刻は `datetime.now(timezone.utc)` で UTC にする** のが安全です。
`isoformat(timespec="seconds")` で `2026-10-18T13:59:08+00:00` のような形式になります（Lambda や TypeScript 側でもそのまま読める）。

日本時間が必要なら `from zoneinfo import ZoneInfo` → `datetime.now(ZoneInfo("Asia/Tokyo"))`（Python 3.9+）。

---

## 14.2 decimal — お金の計算

### 題材コード

`generate_dummy_csv.py:181-183`

```python
def yen_round(value: Decimal) -> int:
    """小数第 1 位で四捨五入して円単位にする(元システム・アプリと同じ規則)。"""
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
```

`generate_dummy_csv.py:336, 352`

```python
tax_rate = None if rng.random() < 0.1 else Decimal("0.1")
...
ln["明細消費税"] = yen_round(Decimal(ln["販売小計(税別)"]) * tax_rate) if tax_rate else 0
```

### 解説

`float`（小数）は 2 進数で近似しているので、**10 進数の小数を正確に表せません**。

```python
>>> 0.1 * 3
0.30000000000000004
>>> round(1234.5)
1234                     # 銀行丸め（偶数への丸め）で切り捨てられる
```

お金の計算でこれは困ります。`decimal.Decimal` は **10 進数のまま正確に計算** できる型です。

```python
>>> from decimal import Decimal, ROUND_HALF_UP
>>> Decimal("0.1") * 3
Decimal('0.3')
>>> Decimal("12345") * Decimal("0.1")
Decimal('1234.5')
>>> (Decimal("12345") * Decimal("0.1")).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
Decimal('1235')          # 正しく四捨五入
```

- **`Decimal("0.1")` と文字列で作る** のが鉄則です。`Decimal(0.1)` と float から作ると、float の誤差ごと取り込んでしまいます
  （`Decimal('0.1000000000000000055511151231257827021181583404541015625')`）。整数からなら `Decimal(12345)` で問題ありません。
- `quantize(Decimal("1"))` は「1 の位まで（小数 0 桁）に揃える」、`quantize(Decimal("0.01"))` なら小数 2 桁。
- `rounding=ROUND_HALF_UP` が **四捨五入**。日本の消費税の端数処理は事業者が決めるので、元システムに合わせて選んでいます。
  切り捨てなら `ROUND_DOWN`、切り上げなら `ROUND_UP`。

docstring の「元システム・アプリと同じ規則」が重要で、dummy_csv が出力した消費税と、アプリ（TypeScript）が計算した消費税を **1 円単位で突き合わせる** ためです。
PHP なら `bcmath` / `brick/money`、JS なら `decimal.js` / `big.js` に当たる役割です。

---

## 14.3 random — 再現できる乱数

### 題材コード

`generate_dummy_csv.py:507`

```python
rng = random.Random(args.seed)
```

`generate_dummy_csv.py:239`（抜粋）

```python
names = rng.sample(SOFTWARE_NAMES, k=rng.choices([1, 2, 3], weights=[6, 3, 1])[0])
```

### 解説

`random.Random(シード)` は **専用の乱数生成器** を作ります。**同じシードなら毎回同じ乱数の並び** になります。

```python
>>> r1, r2 = random.Random(42), random.Random(42)
>>> [r1.randint(1, 6) for _ in range(5)]
[6, 1, 1, 6, 3]
>>> [r2.randint(1, 6) for _ in range(5)]
[6, 1, 1, 6, 3]          # 同じ
```

`--seed` の help に「同じ値なら同じ CSV になる」とあるのはこのためです。テストデータを作るとき、
**毎回同じデータが出る** と「前回と結果が違うのはコードを変えたせいだ」と切り分けられます。

`random.randint(...)` のように **モジュールの関数を直接使う** こともできますが、それはプログラム全体で共有の生成器です。
dummy_csv が `rng` を作って関数に引数で渡している（`make_order(rng, ...)`）のは、**乱数の出どころを 1 つに管理する** ためです。

| メソッド | 意味 | 題材での用途 |
|---|---|---|
| `rng.random()` | 0 以上 1 未満の小数 | `if rng.random() < 0.03:`（3% の確率で） |
| `rng.randint(a, b)` | a 以上 **b 以下** の整数（b を含む） | 日数・秒数 |
| `rng.uniform(a, b)` | a〜b の小数 | 販売単価の倍率 |
| `rng.choice(xs)` | 1 つ選ぶ | 顧客・営業担当 |
| `rng.choices(xs, weights=[...], k=1)` | **重み付き** で k 個選ぶ（重複あり）。**リストで返る** | ステータス・明細数 |
| `rng.sample(xs, k)` | 重複なしで k 個選ぶ | ソフト名 |
| `rng.shuffle(xs)` | その場で並べ替え | |

`rng.choices(...)` は **常にリストを返す** ので、1 個だけほしいときは `[0]` を付けます。dummy_csv のあちこちにある `[0]` はこれです。

`if rng.random() < 0.03:` は「3% の確率で」の定番の書き方です。dummy_csv はこれで「削除フラグ付きの案件」「改行を含む値」などの
**境界のケースを一定の割合で混ぜて**、取り込み処理のテストに使っています。

> **注意**: `random` は暗号用途（パスワードやトークンの生成）には使えません。その場合は `secrets` モジュールを使います。

---

## 14.4 hashlib — ハッシュ値

### 題材コード

`scripts/embedders.py:39-42`

```python
def _cache_path(self, text: str, input_type: str) -> Path:
    raw = f"{self.model}|{self.dim}|{input_type}|{text}"
    key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{self.model}-{self.dim}" / f"{key}.json"
```

### 解説

ハッシュ関数は、どんな長さのデータからも **決まった長さの「指紋」** を作ります。同じ入力からは必ず同じ値、少しでも違えばまったく違う値になります。

- `raw.encode("utf-8")` — ハッシュ関数は **バイト列** を受け取るので、文字列を UTF-8 のバイト列に変換する
- `hashlib.sha256(...)` — SHA-256 で計算
- `.hexdigest()` — 結果を 64 桁の 16 進数の文字列にする（PHP の `hash('sha256', $s)`）

**長い本文をそのままファイル名にはできない** ので、ハッシュ値をファイル名にしています（第 7 章のキャッシュ）。
eval.py の `Router` も、プロンプトと候補の組み合わせを SHA-256 にしてキャッシュのキーにしています。

### ハッシュで「振り分ける」— MockEmbedder

`reference/okubo-shamisen/embedders.py:74-80`

```python
for n in (2, 3):
    for i in range(max(0, len(t) - n + 1)):
        g = t[i : i + n]
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
        v[h % self.dim] += 1.0
norm = sum(x * x for x in v) ** 0.5 or 1.0
vecs.append([x / norm for x in v])
```

API を使わない偽物の埋め込みで、面白い使い方をしています。

1. 文字の 2-gram・3-gram（第 2 章のバイグラムの仲間）を取り出す
2. その MD5 ハッシュの先頭 8 桁を `int(..., 16)` で **16 進数として整数に** する
3. `h % self.dim` で 0〜255 のどれかの位置に振り分け、そこを +1 する
4. 最後にベクトルの長さで割って **長さ 1 に正規化**（`or 1.0` は全部 0 のときの割り算エラー対策）

似た文字列は同じ n-gram を多く含むので、同じ位置が増え、コサイン類似度が高くなります。
**意味は分からないが、文字の重なりは測れる** ベクトルです（「feature hashing」と呼ばれる手法）。
MD5 は安全性の面ではもう使われませんが、このように **ただ均等に振り分けたいだけ** の用途なら問題ありません。

---

## 14.5 iter / next / itertools — イテレータを手で操る

### 題材コード

`generate_dummy_csv.py:530-538`

```python
per_file = -(-len(all_rows) // len(targets))  # 切り上げ
rows_iter = iter(all_rows)
for target in targets:
    with target.open("w", ...) as f:
        writer = csv.writer(f, ...)
        writer.writerow(HEADERS)
        writer.writerows(islice(rows_iter, per_file))
```

`reference/okubo-shamisen/build_kb.py:126-130`

```python
q_vecs = emb.embed(q_texts, "query")
it = iter(q_vecs)
faq_out = []
for e in entries:
    vecs = [round_vec(next(it)) for _ in e.questions]
```

### 解説

`iter(リスト)` は、リストから **「次の 1 個」を順に取り出す道具（イテレータ）** を作ります。
イテレータは **どこまで読んだかを覚えている** のがポイントです。

```python
>>> it = iter([1, 2, 3])
>>> next(it), next(it)
(1, 2)
>>> next(it)
3
>>> next(it)
StopIteration           # もう無い
```

**dummy_csv の分割書き出し**

- `islice(rows_iter, per_file)`（`itertools.islice`）は「イテレータから **次の per_file 個** だけ取り出す」
- 1 つ目のファイルで先頭から per_file 行、**同じ `rows_iter` を使い回す** ので、2 つ目のファイルではその続きから per_file 行…と自然に分かれます
- 最後のファイルは残りが per_file 行に満たなくても、あるだけ書かれます

```python
>>> it = iter(range(10))
>>> [list(islice(it, 4)) for _ in range(3)]
[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
```

添字の計算（`all_rows[i * per_file : (i + 1) * per_file]`）でも書けますが、イテレータを使うと「続きから」が自然に書けます。

**ref_build_kb の対応付け**

全 FAQ の全質問文を **1 回の API 呼び出しでまとめて** 埋め込み（`q_vecs` は平らなリスト）、
それを `next(it)` で **FAQ ごとの質問数ぶんずつ** 取り出して、元の FAQ に戻しています。
「まとめて処理 → 元の構造に戻す」ときによく使うテクニックです。

### itertools のよく使う道具

| 関数 | 意味 |
|---|---|
| `islice(it, n)` | 先頭から n 個 |
| `chain(a, b)` | つなげて 1 本に |
| `groupby(xs, key=)` | 連続する同じキーでまとめる（**先に sorted しておく**） |
| `product(a, b)` | すべての組み合わせ（入れ子の for） |
| `batched(xs, n)` | n 個ずつに区切る（Python 3.12+）。embedders の `range(0, len, BATCH)` + スライスを 1 行で |

---

## 14.6 collections — 便利な入れ物

題材では使っていませんが、第 2 章の「`d.get(k, 0) + 1` で数える」を書く場面では、実務ではこちらもよく使います。

```python
from collections import Counter, defaultdict

# Counter: 数える専用の辞書
>>> Counter(["振込", "振込", "代引き"])
Counter({'振込': 2, '代引き': 1})
>>> Counter(["振込", "振込", "代引き"]).most_common(1)
[('振込', 2)]

# defaultdict: 無いキーを読むと、自動で初期値を作る辞書
>>> by_faq = defaultdict(list)
>>> by_faq["fee"].append("月謝は？")      # "fee" が無くても KeyError にならない
>>> dict(by_faq)
{'fee': ['月謝は？']}
```

textmatch の `bigrams()` は `Counter(text[i:i+2] for i in range(len(text) - 1))` の 1 行でも書けます
（題材が `dict` で書いているのは、JS / PHP に移植するときに同じ形で書けるようにするためと考えられます）。

---

## 14.7 その他、名前だけ覚えておくとよいもの

| モジュール | 用途 |
|---|---|
| `os` / `sys` | 環境変数・終了・コマンドライン引数（第 6・8・13 章） |
| `time` | `time.sleep()` / `time.perf_counter()`（第 12 章） |
| `math` | `math.ceil` / `math.sqrt` / `math.isclose`（小数の比較） |
| `statistics` | `mean` / `median` / `quantiles` — eval.py の `pct()` は `statistics.quantiles` でも書ける |
| `shutil` | ファイル・フォルダのコピーや削除（`shutil.rmtree`） |
| `subprocess` | 外部コマンドの実行 |
| `logging` | ログ出力（第 15 章） |
| `urllib.request` | HTTP（ref_embedders）。実務ではサードパーティの `requests` / `httpx` が主流 |
| `concurrent.futures` | スレッド・プロセスで並列実行 |
| `tempfile` | 一時ファイル・一時フォルダ |

---

## 章末問題

### 問 1（読解）

次の値は？

```python
month_end(date(2026, 2, 3))
date(2026, 10, 18) + timedelta(days=14)
(date(2026, 12, 25) - date(2026, 10, 18)).days
f"{datetime(2025, 10, 2, 8, 10, 40):%Y-%m-%d %H:%M:%S}"
```

### 問 2（書く）

dummy_csv の `yen_round()` を使って、販売小計 `[1980, 2450, 333]` それぞれの消費税（10%、四捨五入）と、その合計を求めてください。
同じ計算を `round(x * 0.1)` でやったときと結果を比べてください。

### 問 3（書く）

`random.Random(7)` を使って、`["物販", "受託", "その他", "対象外"]` から重み `[5, 3, 1, 1]` で 10 件選び、
`Counter` で件数を数えて表示してください。2 回実行して同じ結果になることを確かめてください。

### 問 4（書く）

`embedders.py` の `_cache_path()` を真似て、「質問文」と「モデル名」から SHA-256 の先頭 12 桁を返す関数 `cache_key(question, model)` を書いてください。
`cache_key("月謝は？", "voyage-3.5")` と `cache_key("月謝は？ ", "voyage-3.5")`（末尾にスペース）の結果を比べてください。

### 問 5（書く）

`rows = list(range(1, 11))` を、`iter` と `islice` を使って **3 ファイル分** にほぼ均等に分けたリストのリストにしてください。
1 ファイルあたりの件数は dummy_csv と同じく切り上げで計算します。

### 問 6（読解）

ref_build_kb.py の `next(it)` を使った対応付けで、もし `emb.embed()` が返すベクトルの数が質問文の数より **1 つ少なかった** ら、何が起きますか？

---

## 回答

<details>
<summary>問 1 の回答</summary>

| 式 | 結果 |
|---|---|
| `month_end(date(2026, 2, 3))` | `date(2026, 2, 28)`（2026 年はうるう年ではない） |
| `date(2026, 10, 18) + timedelta(days=14)` | `date(2026, 11, 1)` |
| `(date(2026, 12, 25) - date(2026, 10, 18)).days` | `68` |
| `f"{datetime(2025, 10, 2, 8, 10, 40):%Y-%m-%d %H:%M:%S}"` | `'2025-10-02 08:10:40'` |

</details>

<details>
<summary>問 2 の回答</summary>

```python
from decimal import Decimal, ROUND_HALF_UP


def yen_round(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


subtotals = [1980, 2450, 333]
taxes = [yen_round(Decimal(s) * Decimal("0.1")) for s in subtotals]
print(taxes, sum(taxes))                       # [198, 245, 33] 476

print([round(s * 0.1) for s in subtotals])     # [198, 245, 33]
```

この例では同じ結果ですが、`2445` のように **税額がちょうど .5 になる金額** だと違いが出ます。

```python
>>> yen_round(Decimal(2445) * Decimal("0.1"))
245
>>> round(2445 * 0.1)
244      # 244.5 → 偶数への丸めで 244
```

1 円のずれが、何万件も集計すると大きな差になります。

</details>

<details>
<summary>問 3 の回答</summary>

```python
import random
from collections import Counter

rng = random.Random(7)
picked = rng.choices(["物販", "受託", "その他", "対象外"], weights=[5, 3, 1, 1], k=10)
print(Counter(picked))
```

シードが同じなので、何度実行しても同じ結果になります。`random.Random()` とシードを渡さないと、毎回変わります。

</details>

<details>
<summary>問 4 の回答</summary>

```python
import hashlib


def cache_key(question: str, model: str) -> str:
    raw = f"{model}|{question}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


print(cache_key("月謝は？", "voyage-3.5"))
print(cache_key("月謝は？ ", "voyage-3.5"))   # まったく違う値になる
```

スペース 1 つでも別のキーになるので、キャッシュを効かせたいなら **キーを作る前に正規化（`strip()` など）する** ことを検討します。
ただし embedders.py は「本文が 1 文字でも違えば別のベクトル」なので、正規化しないのが正しい設計です。

</details>

<details>
<summary>問 5 の回答</summary>

```python
from itertools import islice

rows = list(range(1, 11))
n_files = 3
per_file = -(-len(rows) // n_files)   # 切り上げ: 4
it = iter(rows)
parts = [list(islice(it, per_file)) for _ in range(n_files)]
print(parts)   # [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]]
```

</details>

<details>
<summary>問 6 の回答</summary>

最後の FAQ の最後の質問で `next(it)` を呼んだとき、もう値が無いので **`StopIteration`** が発生し、処理が止まります。
**数がずれたまま黙って進むことはありません**。

（細かい話: リスト内包表記 `[... next(it) ...]` の中ならそのまま `StopIteration` が外に出ますが、
ジェネレータ式 `(... next(it) ...)` の中で起きると `RuntimeError: generator raised StopIteration` に変わります。
どちらにしてもエラーで止まります。）

逆に「ベクトルの方が 1 つ多い」場合はエラーにならず、最後の 1 つが使われないまま残ります。
気になるなら、最後に `assert next(it, None) is None`（もう残っていないこと）を確認すると堅くなります。

</details>
