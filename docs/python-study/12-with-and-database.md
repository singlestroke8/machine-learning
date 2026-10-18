# 第 12 章 with 文・コンテキストマネージャ・DB 接続

頻度: ★★★☆☆ — ファイル・DB・ネットワーク接続を扱うコードには必ず出てきます。

この章で読めるようになるコード:

```python
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    register_vector(conn)
    with conn.transaction():
        deleted = conn.execute("DELETE FROM chunks WHERE model = %s", (emb.model,)).rowcount
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, [...])
```

---

## 12.1 with 文は「後片付けの自動化」

第 7 章で `with target.open(...) as f:` を見ました。`with` の働きをあらためて整理します。

```python
with 何か as 変数:
    処理
# ← ブロックを抜けた時点で「何か」の後片付けが必ず行われる
```

「後片付け」の中身は、`with` に渡したものによって決まります。

| with に渡すもの | 入るとき | 抜けるとき |
|---|---|---|
| `open(...)` | ファイルを開く | **ファイルを閉じる** |
| `psycopg.connect(...)` | DB に接続 | **コミット（例外ならロールバック）して接続を閉じる** |
| `conn.transaction()` | トランザクション開始 | **コミット（例外ならロールバック）** |
| `conn.cursor()` | カーソルを作る | **カーソルを閉じる** |
| `urllib.request.urlopen(...)` | HTTP 接続 | **接続を閉じる** |

**ブロックの途中で例外が起きても、後片付けは必ず実行されます**。これが `with` を使う最大の理由です。

> **PHP / JS との比較**
>
> ```php
> $pdo->beginTransaction();
> try {
>     $pdo->exec(...);
>     $pdo->commit();
> } catch (Throwable $e) {
>     $pdo->rollBack();
>     throw $e;
> }
> ```
>
> PHP で書くこの try / catch / rollBack を、Python では `with conn.transaction():` の 1 行で書けます。
> Laravel の `DB::transaction(function () { ... })` に近い感覚です。
> JS（TypeScript 5.2+）の `using` 宣言が、同じ考え方を後から取り入れたものです。

---

## 12.2 load_kb.py を読む — 入れ子の with

### 題材コード

`scripts/load_kb.py:44-58`

```python
# 3. DB に接続する。DATABASE_URL は docker-compose.yml が py コンテナに渡している
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    # vector 型と numpy 配列を相互変換できるようにする
    register_vector(conn)

    # 4. 削除と投入を 1 つのトランザクションにまとめる。途中で失敗したら丸ごと元に戻る
    with conn.transaction():
        deleted = conn.execute("DELETE FROM chunks WHERE model = %s", (emb.model,)).rowcount
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, [
                (c["page_key"], c["title"], c["heading"] or None, c["text"], emb.model, np.array(v))
                for c, v in zip(chunks, vecs)
            ])

    # 5. 投入後の件数をモデル別に数えて表示する
    rows = conn.execute("SELECT model, count(*) FROM chunks GROUP BY model ORDER BY model").fetchall()
```

### 解説

with が 3 段の入れ子になっています。外側から順に見ます。

**① `with psycopg.connect(...) as conn:` — 接続**

- psycopg（PostgreSQL 用のライブラリ、バージョン 3）で DB に接続し、接続オブジェクトを `conn` に入れます。
- 接続先の URL は環境変数から（第 6 章）。パスワードをコードに書かないためです。
- ブロックを抜けると、接続が **自動で閉じられます**。

**② `register_vector(conn)` — 型の登録**

pgvector の `vector` 型と NumPy の配列を、Python と DB の間で自動変換できるようにする設定です。
これで `np.array(v)` をそのままパラメータとして渡せます。

**③ `with conn.transaction():` — トランザクション**

- 「同じモデルの古い行を全部消す」→「新しい行を全部入れる」を **1 つのトランザクション** にしています。
- 投入の途中で失敗すると、**削除も含めて丸ごと取り消され** ます（ロールバック）。
- これが無いと、「削除は成功・投入は途中で失敗」という **データが半分消えた状態** が起きえます。
  コメントの「途中で失敗したら丸ごと元に戻る」は、まさにこのことです。

**④ `with conn.cursor() as cur:` — カーソル**

`executemany` のために明示的にカーソルを作り、使い終わったら閉じます。
psycopg 3 では `conn.execute(...)` と書くと、内部でカーソルを作って実行し、そのカーソルを返してくれる **ショートカット** があります。
`DELETE` と `SELECT` はそのショートカットを使っています。

### rowcount と fetchall

```python
deleted = conn.execute("DELETE ...", (emb.model,)).rowcount       # 影響した行数
rows = conn.execute("SELECT model, count(*) ...").fetchall()       # 全行をリストで
```

| メソッド / 属性 | 戻り値 |
|---|---|
| `cur.fetchall()` | 全行の **タプルのリスト** `[("voyage-3.5", 412), ("voyage-3.5-lite", 412)]` |
| `cur.fetchone()` | 1 行のタプル（無ければ `None`） |
| `cur.rowcount` | `DELETE` / `UPDATE` / `INSERT` で影響した行数 |
| `for row in cur:` | 1 行ずつ（大量の行をメモリに載せずに読める） |

行がタプルで返るので、受け取る側でアンパックします（`for model, n in rows:`、export_kb.py の内包表記）。
列名で読みたい場合は `psycopg.connect(url, row_factory=psycopg.rows.dict_row)` で **辞書の行** にできます。

---

## 12.3 SQL のパラメータ化

### 題材コード

`scripts/load_kb.py:26-29`

```python
INSERT_SQL = """
    INSERT INTO chunks (page_key, title, heading, text, model, embedding)
    VALUES (%s, %s, %s, %s, %s, %s)
"""
```

### 解説

`%s` は **プレースホルダ** です。値は SQL の文字列に埋め込まず、`execute(SQL, パラメータのタプル)` の第 2 引数で別に渡します。
ライブラリが値を安全に（エスケープして、または DB に別送して）扱ってくれるので、**SQL インジェクションを防げます**。

```python
# 正しい
conn.execute("DELETE FROM chunks WHERE model = %s", (emb.model,))

# 絶対にダメ（f 文字列で値を埋め込む）
conn.execute(f"DELETE FROM chunks WHERE model = '{emb.model}'")
```

f 文字列の方は、`emb.model` に `x' OR '1'='1` のような文字列が入ると **全行削除** になります。

| ライブラリ | プレースホルダ |
|---|---|
| psycopg（PostgreSQL） | `%s`（型に関係なく常に `%s`）、名前付きは `%(name)s` |
| sqlite3（標準ライブラリ） | `?`、名前付きは `:name` |
| PHP の PDO | `?`、名前付きは `:name` |

psycopg の `%s` は Python の古い文字列書式 `"%s" % x` と見た目が同じですが、**まったく別物** です。
`INSERT_SQL % values` のように自分で `%` 演算子を使って埋め込んではいけません。

### executemany — まとめて INSERT

```python
cur.executemany(INSERT_SQL, [ (1 行目の値...), (2 行目の値...), ... ])
```

同じ SQL を、パラメータのリストの各要素で繰り返し実行します。for 文で `execute` を何百回も呼ぶより速く、コードも短くなります。
第 2 引数はリスト内包表記で、チャンクとベクトルを `zip` して 1 行分のタプルを作っています（第 5 章）。

`c["heading"] or None` で空文字を `None` に変えると、DB には **`NULL`** として入ります（第 1 章）。
`db/init/` のテーブル定義でも、`heading` 列だけ `NOT NULL` が付いていません。

> PivotAggregationSystem の CLAUDE.md にある「SQL の値は必ずパラメータ化する。テーブル名・列名は … `ident()` を通して埋め込む」も同じ考え方です。
> **値はパラメータで、識別子（テーブル名・列名）は検査済みのものだけを専用の関数で** — 言語が変わっても変わらない鉄則です。

---

## 12.4 export_kb.py — 接続を早く閉じる

### 題材コード

`scripts/export_kb.py:35-47`

```python
# 1. DB から指定モデルの行を id 順に読む
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    # embedding::text は "[0.01,-0.02,...]" という文字列で返り、そのまま JSON として読める。
    # pgvector ライブラリの版によって戻り値の型（numpy 配列 / Vector）が違うのを避けるため
    rows = conn.execute(
        "SELECT id, page_key, title, heading, text, embedding::text"
        " FROM chunks WHERE model = %s ORDER BY id",
        (args.model,),
    ).fetchall()

# 2. 0 件なら load_kb.py の実行忘れなので止める（空の kb.json を Lambda に載せない）
if not rows:
    raise SystemExit(...)
```

### 解説

- `with` のブロックには **DB から読む処理だけ** を入れ、`fetchall()` で全行を手元に取ったら、すぐブロックを抜けて接続を閉じています。
  JSON への変換やファイルの書き出しは、接続を閉じた後に行います。**接続は必要な間だけ持つ** のが行儀の良い書き方です。
- `with` のブロックの中で作った変数（`rows`）は、**ブロックを抜けた後も使えます**。
  Python の `with` / `if` / `for` はスコープを作らないからです（JS の `let` のブロックスコープとは違う）。
- `embedding::text` は PostgreSQL のキャスト構文。ベクトルを文字列で受け取り、`json.loads()` で float のリストにしています（第 5 章）。
  **ライブラリのバージョン差を避けるために、あえて単純な型で受け取る** という実務的な判断です。

---

## 12.5 自分でコンテキストマネージャを作る

`with` に渡せるものは自分でも作れます。実務でよく作るのは「処理時間の計測」です。

### contextlib.contextmanager を使う方法（おすすめ）

```python
import time
from contextlib import contextmanager


@contextmanager
def timer(label: str):
    """with ブロックの処理時間をミリ秒で表示する。"""
    start = time.perf_counter()          # ← with に入るときに実行される
    try:
        yield                            # ← ここで with ブロックの中身が実行される
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"{label}: {elapsed_ms:.0f} ms")   # ← 抜けるときに必ず実行される


with timer("埋め込み"):
    vecs = emb.embed(texts, "document")
# 埋め込み: 834 ms
```

- `@contextmanager` を付けた関数の **`yield` の前が「入るとき」、後が「抜けるとき」** の処理になります。
- `try` / `finally` で囲んでおくと、ブロック内で例外が起きても時間が表示されます。
- `time.perf_counter()` は処理時間の計測専用の時計です（`time.time()` は時刻用で、計測には向かない）。

このリポジトリの費用ログ（CLAUDE.md の `ttft_ms` / `total_ms`）のような計測を、Python 側でやりたくなったときに使える形です。

### クラスで作る方法（仕組みを知るため）

```python
class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):                  # with に入るとき。戻り値が as の変数に入る
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):   # 抜けるとき。例外の情報が渡される（無ければ None）
        self.ms = (time.perf_counter() - self.start) * 1000
        print(f"{self.label}: {self.ms:.0f} ms")
        return False                      # False: 例外は握りつぶさず、そのまま外に伝える


with Timer("DB 投入") as t:
    ...
print(t.ms)                               # ブロックの後でも結果を読める
```

`with X as y:` は、内部では `y = X.__enter__()` → ブロック → `X.__exit__(...)` という呼び出しです（第 10 章の特殊メソッド）。
psycopg の接続もファイルも、この 2 つのメソッドを持っているから `with` で使えるのです。

---

## 12.6 手元で試す — sqlite3（標準ライブラリ）

PostgreSQL が無くても、標準ライブラリの `sqlite3` で同じ考え方を試せます。

```python
import sqlite3

conn = sqlite3.connect(":memory:")     # メモリ上の使い捨て DB
conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, model TEXT NOT NULL, text TEXT NOT NULL)")

rows = [("voyage-3.5-lite", "月謝は8000円"), ("voyage-3.5-lite", "体験あり"), ("voyage-3.5", "体験あり")]
with conn:                             # sqlite3 の接続を with に渡すとトランザクションになる
    conn.executemany("INSERT INTO chunks (model, text) VALUES (?, ?)", rows)

print(conn.execute("SELECT model, count(*) FROM chunks GROUP BY model ORDER BY model").fetchall())
# [('voyage-3.5', 1), ('voyage-3.5-lite', 2)]
conn.close()
```

> **注意: ライブラリによって with の意味が違う**
> sqlite3 の接続を `with` に渡すと **コミット / ロールバックだけ** を行い、**接続は閉じません**。
> psycopg 3 は **接続も閉じます**。「with が何をしてくれるか」はライブラリのドキュメントで確認する習慣を付けましょう。

---

## 章末問題

### 問 1（読解）

load_kb.py で、`executemany` の途中（例えば 300 行目）で DB のエラーが起きたとします。
DB の `chunks` テーブルはどういう状態になりますか？ `with conn.transaction():` が無かった場合はどうなりますか？

### 問 2（バグ探し）

次のコードの問題点を指摘し、直してください。

```python
def delete_model(conn, model: str) -> int:
    return conn.execute(f"DELETE FROM chunks WHERE model = '{model}'").rowcount
```

### 問 3（書く・sqlite3 で）

12.6 の `chunks` テーブルに対して、「指定したモデルの行を全部消して、新しい行のリストを入れる」関数
`replace_model(conn, model, texts)` を書いてください。削除と投入は 1 つのトランザクションにし、
消した行数と入れた行数を返してください。

### 問 4（実験）

問 3 の関数に、`texts` の中に `None` を 1 つ混ぜて渡してください（`text` 列は `NOT NULL`）。
エラーの後、テーブルの中身はどうなっていますか？

### 問 5（書く）

`@contextmanager` を使って、ブロックの処理時間を **リストに記録する** コンテキストマネージャ `record(log: list, label: str)` を書いてください。
ブロックを抜けると `log` に `{"label": ..., "ms": ...}` が追加されるようにします。

---

## 回答

<details>
<summary>問 1 の回答</summary>

- **transaction がある場合**: 例外で `with conn.transaction():` を抜けるときにロールバックされ、
  `DELETE` も途中までの `INSERT` も取り消されます。テーブルは **実行前とまったく同じ状態** です。
- **無い場合**: psycopg 3 は接続の `with` を例外で抜けるときにもロールバックするので、この単純なスクリプトでは結果的に同じになります。
  ただし、途中で `conn.commit()` を呼ぶコードや、自動コミット（`autocommit=True`）の接続では、
  **古い行は消えたのに新しい行は 299 行だけ** という壊れた状態が残ります。
  「ここからここまでは 1 つの操作」と **意図をコードで明示する** ためにも、`transaction()` を書くのが正解です。

</details>

<details>
<summary>問 2 の回答</summary>

f 文字列で値を SQL に埋め込んでいるため、**SQL インジェクション** の危険があります。
`model` に `' OR '1'='1` が渡されると `WHERE model = '' OR '1'='1'` となり、全行が消えます。

```python
def delete_model(conn, model: str) -> int:
    return conn.execute("DELETE FROM chunks WHERE model = %s", (model,)).rowcount
```

</details>

<details>
<summary>問 3 の回答</summary>

```python
import sqlite3


def replace_model(conn: sqlite3.Connection, model: str, texts: list[str]) -> tuple[int, int]:
    with conn:   # 削除と投入を 1 つのトランザクションに
        deleted = conn.execute("DELETE FROM chunks WHERE model = ?", (model,)).rowcount
        conn.executemany(
            "INSERT INTO chunks (model, text) VALUES (?, ?)",
            [(model, t) for t in texts],
        )
    return deleted, len(texts)


conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, model TEXT NOT NULL, text TEXT NOT NULL)")
print(replace_model(conn, "lite", ["a", "b"]))        # (0, 2)
print(replace_model(conn, "lite", ["c", "d", "e"]))   # (2, 3)
```

load_kb.py とまったく同じ「全削除 → 全投入」の形です。

</details>

<details>
<summary>問 4 の回答</summary>

```python
try:
    replace_model(conn, "lite", ["x", None])
except sqlite3.IntegrityError as e:
    print(e)   # NOT NULL constraint failed: chunks.text
print(conn.execute("SELECT text FROM chunks WHERE model = 'lite'").fetchall())
# [('c',), ('d',), ('e',)]
```

`DELETE` も `"x"` の `INSERT` もロールバックされ、**前回の 3 行がそのまま残っています**。
トランザクションのおかげで、データが中途半端な状態になりません。

</details>

<details>
<summary>問 5 の回答</summary>

```python
import time
from contextlib import contextmanager


@contextmanager
def record(log: list, label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        log.append({"label": label, "ms": round((time.perf_counter() - start) * 1000)})


log: list[dict] = []
with record(log, "検索"):
    time.sleep(0.03)
with record(log, "生成"):
    time.sleep(0.05)
print(log)   # [{'label': '検索', 'ms': 30}, {'label': '生成', 'ms': 50}]（環境により数 ms ずれる）
```

この形でリクエストごとの段階別の時間を集めておくと、後で `json.dumps` して 1 行 JSON のログに書き出せます。

</details>
