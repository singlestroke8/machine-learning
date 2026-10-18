# 第 10 章 クラスとオブジェクト指向

頻度: ★★★☆☆ — 自分で書く回数は関数より少ないですが、ライブラリを使うときは常にクラスを触っています。
PHP（Laravel）でクラスに慣れているなら、違いさえ押さえればすぐ書けます。

この章で読めるようになるコード:

```python
class Embedder:
    def __init__(self, model: str = DEFAULT_MODEL, dim: int = DIM):
        self.model = model

class MockEmbedder(_CachedEmbedder):
    provider = "mock"

    def _embed_batch(self, texts, input_type): ...

def __call__(self, question: str, cands: list[dict]) -> str | None:
```

---

## 10.1 クラスの基本 — Embedder を読む

### 題材コード

`scripts/embedders.py:27-44`（抜粋）

```python
class Embedder:
    def __init__(self, model: str = DEFAULT_MODEL, dim: int = DIM):
        self.model = model
        self.dim = dim
        # API キーは環境変数 VOYAGE_API_KEY から読まれる（make py が .env を渡す）。
        # 429 や 5xx は SDK が指数バックオフで再試行する。
        self.client = voyageai.Client(max_retries=3, timeout=60)
        self.billed_tokens = 0  # このプロセスで実際に課金されたトークン数
        self.cache_hits = 0

    def _cache_path(self, text: str, input_type: str) -> Path:
        raw = f"{self.model}|{self.dim}|{input_type}|{text}"
        ...

    def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        ...
        self.billed_tokens += res.total_tokens
```

使う側（`scripts/embedders.py:99-104`）

```python
emb = Embedder(args.model)
vecs = emb.embed([chunk_to_text(c) for c in chunks], "document")
print(f"model: {emb.model} / dim: {emb.dim}")
print(f"chunks: {len(vecs)} / cache hits: {emb.cache_hits} / billed tokens: {emb.billed_tokens}")
```

### 解説

| Python | PHP | JS | 意味 |
|---|---|---|---|
| `class Embedder:` | `class Embedder {` | `class Embedder {` | クラスの定義 |
| `def __init__(self, ...)` | `public function __construct(...)` | `constructor(...)` | コンストラクタ |
| `self.model = model` | `$this->model = $model;` | `this.model = model;` | インスタンスの属性 |
| `def embed(self, ...)` | `public function embed(...)` | `embed(...)` | メソッド |
| `Embedder("voyage-3.5")` | `new Embedder("voyage-3.5")` | `new Embedder("voyage-3.5")` | インスタンス化（**`new` は無い**） |
| `emb.embed(...)` | `$emb->embed(...)` | `emb.embed(...)` | メソッド呼び出し |

**PHP / JS との大きな違いが 3 つあります。**

1. **`self` をメソッドの第 1 引数に必ず書く**。
   呼び出すときは `emb.embed(texts, "document")` と `self` を渡しませんが、Python が自動で `emb` を第 1 引数に入れます。
   名前は慣習で `self`（PHP / JS の `this` に当たる）。書き忘れると `TypeError: embed() takes 2 positional arguments but 3 were given` という
   一見意味不明なエラーになります。
2. **属性（プロパティ）を宣言しない**。`__init__` の中で `self.billed_tokens = 0` と **代入した時点で生まれます**。
   PHP のように `private int $billedTokens;` を並べる必要はありません。
   そのため「このクラスにどんな属性があるか」は `__init__` を読んで把握します。属性は `__init__` の中で全部初期化しておくのがマナーです。
3. **`public` / `private` が無い**。すべて外から触れます。

### `_` で始まる名前 = 「内部用」

`_cache_path` は先頭に `_` が付いています。これは「**クラスの外から呼ばないでね**」という **慣習** です。
呼べてしまいますが、リンターやエディタが注意してくれます。

| PHP | Python |
|---|---|
| `public` | 普通の名前 `embed` |
| `protected` / `private` | `_embed_batch`（`_` 1 つ。慣習） |
| — | `__secret`（`_` 2 つ。名前が自動で変えられ、サブクラスからも触りにくくなる。めったに使わない） |

### クラスが「状態」を持つ利点

`Embedder` は `billed_tokens`（課金されたトークン数）と `cache_hits`（キャッシュから読めた数）を
**インスタンスの中に溜めていきます**。`embed()` を何度呼んでも、最後に合計を表示できます。

関数だけで同じことをすると、カウンタを毎回引数で渡して戻り値で受け取る必要があり、煩雑になります。
**「設定（model, dim）+ 接続（client）+ 集計（billed_tokens）」をひとまとめに持ち回りたい** のが、ここでクラスを使う理由です。

---

## 10.2 継承とテンプレートメソッド — 本番版の Embedder 群

本番ボット版の `reference/okubo-shamisen/embedders.py` は、**継承** を使って 2 種類の埋め込みを実装しています。

### 題材コード

`reference/okubo-shamisen/embedders.py:24-56`（抜粋）

```python
class _CachedEmbedder:
    provider = "base"
    model = ""
    dim = 0

    def _key(self, text: str, input_type: str) -> str:
        raw = f"{self.provider}|{self.model}|{self.dim}|{input_type}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        """texts をベクトル化して返す。input_type は 'query' か 'document'。"""
        out: list[list[float] | None] = [None] * len(texts)
        misses: list[int] = []
        for i, t in enumerate(texts):
            ...                          # キャッシュにあれば読む
        if misses:
            vecs = self._embed_batch([texts[i] for i in misses], input_type)   # ← ここだけサブクラス任せ
            ...                          # キャッシュに保存
        return out  # type: ignore[return-value]

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        raise NotImplementedError
```

`reference/okubo-shamisen/embedders.py:59-69, 84-97`（抜粋）

```python
class MockEmbedder(_CachedEmbedder):
    """文字2〜3gramをハッシュでdim次元に畳み込むTFベクトル（L2正規化）。"""

    provider = "mock"
    model = "char-ngram-hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        ...                              # API を使わず、文字 n-gram から擬似的なベクトルを作る


class VoyageEmbedder(_CachedEmbedder):
    provider = "voyage"
    ENDPOINT = "https://api.voyageai.com/v1/embeddings"
    BATCH = 100

    def __init__(self, model: str = "voyage-3.5-lite", dim: int = 256):
        ...

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        ...                              # Voyage の API を呼ぶ（リトライ付き）
```

### 解説

`class MockEmbedder(_CachedEmbedder):` の **カッコの中が親クラス** です（PHP の `extends`）。

この 3 つのクラスの役割分担は次のとおりです。

```
_CachedEmbedder（親）
├── embed()          … キャッシュを見て、無いものだけ _embed_batch() に頼み、結果を保存する【共通の流れ】
└── _embed_batch()   … raise NotImplementedError（「サブクラスで書いてね」）
      ├── MockEmbedder._embed_batch()    … 文字 n-gram でベクトルを作る（無料・API 不要・開発用）
      └── VoyageEmbedder._embed_batch()  … Voyage API を呼ぶ（本番用）
```

- **共通の手順（キャッシュの読み書き）は親に 1 回だけ書き、違う部分（実際のベクトル化）だけをサブクラスで書く**。
  これを **テンプレートメソッドパターン** と呼びます。Laravel のフレームワークのクラスを継承して一部だけ上書きするのと同じ発想です。
- 親の `embed()` の中の `self._embed_batch(...)` は、**実際のインスタンスのクラスのメソッド** が呼ばれます。
  `MockEmbedder` のインスタンスなら Mock 版、`VoyageEmbedder` なら Voyage 版。
- `raise NotImplementedError` は「このメソッドはサブクラスで実装してください」という印です。
  親クラスを直接使って `embed()` を呼ぶと、ここでエラーになります。

> **PHP との比較**
> PHP なら `abstract class CachedEmbedder` と `abstract protected function embedBatch(...)` で
> 「実装し忘れたらエラー」を言語が保証してくれます。
> Python で同じことをするには `abc` モジュールを使います（10.6）。題材は `NotImplementedError` で済ませる簡易な方法です。

### 開発用の偽物（Mock）を用意する意味

`MockEmbedder` は API キーも課金もなしで動く「偽物」です。docstring に書かれているとおり、

- パイプライン全体が通しで動くかの確認
- CI（自動テスト）での実行

に使い、最終的な精度の評価だけ本物の Voyage で行います。
**外部サービスに依存する部分を差し替え可能にしておく** のは、テストしやすいコードの基本です。
同じ `embed()` を持っているので、呼び出し側（eval.py）はどちらのクラスかを気にせず使えます。

---

## 10.3 クラス属性とインスタンス属性

### 解説

`provider = "mock"` のように、**クラスの直下（メソッドの外）で代入した変数** はクラス属性です。
**そのクラスのすべてのインスタンスで共有** されます。

```python
class _CachedEmbedder:
    provider = "base"     # クラス属性
    dim = 0

class MockEmbedder(_CachedEmbedder):
    provider = "mock"     # 親のクラス属性を上書き

    def __init__(self, dim: int = 256):
        self.dim = dim    # インスタンス属性（このインスタンスだけの値）
```

```python
>>> m = MockEmbedder()
>>> m.provider          # インスタンスに無い → クラス MockEmbedder を見る
'mock'
>>> m.dim               # インスタンスにある → それを使う
256
>>> MockEmbedder.dim    # クラスの方は 0 のまま
0
```

`self.xxx` を読むと、**インスタンス → そのクラス → 親クラス** の順に探します。

- 親の `_key()` は `self.provider` を使っているので、Mock なら `"mock"`、Voyage なら `"voyage"` が入り、
  **キャッシュのキーがプロバイダごとに分かれます**。
- `ENDPOINT` や `BATCH` のような **クラスごとに決まった定数** をクラス属性にするのは良い使い方です。

### 落とし穴: クラス属性に変更できるものを置く

```python
class Counter:
    items = []            # すべてのインスタンスで 1 つのリストを共有してしまう

a, b = Counter(), Counter()
a.items.append(1)
print(b.items)            # [1] ← b も変わる！
```

第 4 章の「既定値にリストを置かない」と同じ理由です。インスタンスごとの値は **`__init__` の中で `self.items = []`** と初期化します。

> PHP の `static $items` が近いです。PHP では `static` と明示しますが、Python は「クラス直下に書いたらクラス属性」なので気付きにくいのです。

---

## 10.4 ファクトリ関数

### 題材コード

`reference/okubo-shamisen/embedders.py:130-135`

```python
def make_embedder(provider: str, model: str | None, dim: int):
    if provider == "voyage":
        return VoyageEmbedder(model or "voyage-3.5-lite", dim)
    if provider == "mock":
        return MockEmbedder(dim)
    raise SystemExit(f"未知のprovider: {provider}")
```

### 解説

コマンドラインの `--provider voyage` / `--provider mock` という **文字列から、適切なクラスのインスタンスを作る** 関数です。
呼び出し側（build_kb, eval）は `make_embedder(args.provider, ...)` と書くだけで、どのクラスを使うかの判断をここに任せられます。

Laravel のサービスコンテナでインターフェースに実装を紐づける（`bind`）のを、手書きの関数でやっているイメージです。

---

## 10.5 `__call__` — インスタンスを関数のように呼ぶ

### 題材コード

`reference/okubo-shamisen/eval.py:71-90`（抜粋）

```python
class Router:
    """engine.php の route() と同じ指示文・同じ入力形式で、候補からFAQを選ばせる。

    結果は rag/.cache/router/ にキャッシュし、同じ質問・候補での再課金を防ぐ。
    """

    def __init__(self, model: str, school_name: str):
        ...
        self.client = anthropic.Anthropic()
        self.model = model
        self.system = ROUTER_PROMPT.read_text(encoding="utf-8").replace("{{SCHOOL}}", school_name)
        self.calls = 0
        self.cache_dir = RAG_DIR / ".cache/router"

    def __call__(self, question: str, cands: list[dict]) -> str | None:
        ...
```

使う側（`eval.py:142`）

```python
picked = router(question, [by_id[i] for i in cand_ids])
```

### 解説

`__call__` という名前のメソッドを定義すると、**インスタンスを関数のように `router(...)` と呼べる** ようになります。

- `Router` は「API クライアント・システムプロンプト・呼び出し回数・キャッシュ先」という **状態を持った関数** です。
- `decide()` 側は `router` を「質問と候補を渡すと id を返す、呼べるもの」としてだけ扱います。
  テストでは、普通の関数 `lambda q, c: "fee"` を `router` として渡すこともできます（第 4 章「関数は値」）。

> **PHP との比較**
> PHP の **`__invoke`** とまったく同じです（PHP の `__call` は「存在しないメソッドが呼ばれたとき」用で、別物なので注意）。

### 特殊メソッド（ダンダーメソッド）

`__init__` や `__call__` のように前後に `__` が付いたメソッドを **特殊メソッド**（double underscore から「ダンダー」）と呼びます。
Python の組み込みの動作に、自分のクラスを対応させるためのものです。

| メソッド | いつ呼ばれるか | PHP |
|---|---|---|
| `__init__(self, ...)` | `Cls(...)` でインスタンスを作るとき | `__construct` |
| `__call__(self, ...)` | `obj(...)` | `__invoke` |
| `__repr__(self)` | REPL での表示・デバッグ用の文字列 | — |
| `__str__(self)` | `str(obj)` / `print(obj)` | `__toString` |
| `__len__(self)` | `len(obj)` | `Countable::count` |
| `__eq__(self, other)` | `obj == other` | — |
| `__enter__` / `__exit__` | `with obj:`（第 12 章） | — |
| `__iter__` | `for x in obj:` | `IteratorAggregate` |

`__repr__` を定義しておくと、デバッグで `print(emb)` したときに中身が見えて便利です。

```python
class Embedder:
    ...
    def __repr__(self) -> str:
        return f"Embedder(model={self.model!r}, dim={self.dim})"

>>> Embedder()
Embedder(model='voyage-3.5-lite', dim=256)
```

（`!r` は f 文字列で `repr()` を使う指定。文字列に `'` が付きます。）

---

## 10.6 もっと実務的な書き方

### super() — 親の処理を呼ぶ

題材の `MockEmbedder.__init__` は親の `__init__` を呼んでいません（親に `__init__` が無いので問題ない）。
親にも初期化処理がある場合は `super().__init__(...)` で呼びます（PHP の `parent::__construct()`）。

```python
class LoggingEmbedder(Embedder):
    def __init__(self, model: str = DEFAULT_MODEL, dim: int = DIM):
        super().__init__(model, dim)      # 親の初期化（client や billed_tokens の準備）
        self.calls = 0

    def embed(self, texts, input_type):
        self.calls += 1
        return super().embed(texts, input_type)   # 親の embed をそのまま使う
```

### 抽象基底クラス（abc）

```python
from abc import ABC, abstractmethod


class CachedEmbedder(ABC):
    @abstractmethod
    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        """サブクラスで実装する。"""
```

- `@abstractmethod` が付いたメソッドを実装していないサブクラスは、**インスタンスを作った時点で TypeError** になります。
- `NotImplementedError` 方式は「呼んだときに初めてエラー」なので、`abc` の方が早く気付けます。

`@` で始まる行は **デコレータ** で、関数やクラスに機能を足す仕組みです。ほかによく見るものに次があります。

| デコレータ | 意味 |
|---|---|
| `@property` | メソッドを `obj.total` のように属性として読めるようにする（PHP の getter に当たる） |
| `@staticmethod` | `self` を受け取らないメソッド |
| `@classmethod` | インスタンスではなくクラスを第 1 引数 `cls` で受け取るメソッド（別のコンストラクタを作るときに使う） |
| `@dataclass` | 第 15 章 |

### 継承よりも合成

題材の再実装版 `scripts/embedders.py` は、本番版の 3 クラス構成をやめて **`Embedder` 1 クラス** にしています。
Mock が不要になり（Voyage の SDK とキャッシュで十分）、リトライも SDK に任せたからです。

継承は強力ですが、階層が深くなると読みにくくなります。実務では
「**必要になるまで継承しない。差し替えたい部品は引数で受け取る（合成）**」という方針が好まれます。
`decide()` が `router` を引数で受け取っているのは、まさに合成の例です。

---

## 章末問題

### 問 1（読解）

次のコードの出力は？

```python
class Base:
    provider = "base"
    def label(self):
        return f"[{self.provider}] {self.name()}"
    def name(self):
        raise NotImplementedError

class Mock(Base):
    provider = "mock"
    def name(self):
        return "char-ngram"

print(Mock().label())
print(Base().provider)
print(Base().label())
```

### 問 2（バグ探し）

次のクラスを使うとどうなりますか？ 直してください。

```python
class Stats:
    def __init__(self):
        hits = 0
    def add(self):
        self.hits += 1
```

### 問 3（書く）

`textmatch.py` の `bigram_cosine()` を使って、FAQ のリストを持ち、質問に一番近い FAQ を返すクラス `Tier0Matcher` を書いてください。

- `Tier0Matcher(faq, threshold=0.85)` で作る（`faq` は `[{"id": "fee", "questions": ["月謝は？", ...]}, ...]`）
- `matcher.match(question)` で、類似度が閾値以上の最良の FAQ の id を返す（無ければ `None`）
- 何回 `match` が呼ばれたかを `calls` 属性で数える

`bigram_cosine` は自分で import できない環境なら、とりあえず `def bigram_cosine(a, b): return 1.0 if a == b else 0.0` で代用して構いません。

### 問 4（書く）

問 3 のクラスに `__call__` を足して、`matcher(question)` でも `match` と同じ結果が返るようにしてください。
さらに `__repr__` を足して、`Tier0Matcher(faq=2件, threshold=0.85)` と表示されるようにしてください。

### 問 5（考える）

本番版は `_CachedEmbedder` → `MockEmbedder` / `VoyageEmbedder` の継承構成、再実装版は `Embedder` 1 クラスです。
それぞれの良い点を 1 つずつ挙げてください。

---

## 回答

<details>
<summary>問 1 の回答</summary>

```
[mock] char-ngram
base
```

3 行目の `Base().label()` で `NotImplementedError` が投げられ、トレースバックが表示されて止まります。
`self.provider` / `self.name()` は **実際のインスタンスのクラス** から探されるので、`Mock` のインスタンスでは Mock の値・メソッドが使われます。

</details>

<details>
<summary>問 2 の回答</summary>

`__init__` の中の `hits = 0` は **ただのローカル変数** で、インスタンスの属性になりません。
`add()` を呼ぶと `self.hits` が無いので `AttributeError: 'Stats' object has no attribute 'hits'` になります。

```python
class Stats:
    def __init__(self):
        self.hits = 0
    def add(self):
        self.hits += 1
```

PHP で `$this->` を付け忘れるのと同じミスです。

</details>

<details>
<summary>問 3 の回答</summary>

```python
from textmatch import bigram_cosine


class Tier0Matcher:
    def __init__(self, faq: list[dict], threshold: float = 0.85):
        self.faq = faq
        self.threshold = threshold
        self.calls = 0

    def match(self, question: str) -> str | None:
        self.calls += 1
        best_id, best_s = None, 0.0
        for e in self.faq:
            for q in e["questions"]:
                s = bigram_cosine(question, q)
                if s > best_s:
                    best_id, best_s = e["id"], s
        return best_id if best_s >= self.threshold else None


faq = [
    {"id": "fee", "questions": ["月謝はいくらですか？"]},
    {"id": "trial", "questions": ["体験レッスンはありますか？"]},
]
m = Tier0Matcher(faq)
print(m.match("月謝はいくらですか"))     # fee（記号は正規化で消えるので一致）
print(m.match("駐車場はありますか？"))   # None
print(m.calls)                            # 2
```

eval.py の `best_faq_match_t0()` の処理を、閾値と呼び出し回数を持つクラスにまとめた形です。

</details>

<details>
<summary>問 4 の回答</summary>

```python
    def __call__(self, question: str) -> str | None:
        return self.match(question)

    def __repr__(self) -> str:
        return f"Tier0Matcher(faq={len(self.faq)}件, threshold={self.threshold})"
```

</details>

<details>
<summary>問 5 の回答例</summary>

- **継承構成（本番版）**: API キーが無くても Mock で全体を動かせる。CI や開発で課金なしに通しのテストができ、
  埋め込みの提供元を増やすとき（OpenAI など）も `_embed_batch` を書くだけで済む。
- **1 クラス（再実装版）**: 読むファイル・クラスが少なく、処理の流れを上から下まで 1 か所で追える。
  リトライを SDK に任せたぶんコードも短い。Phase の範囲では Voyage しか使わないので、拡張性より単純さを選んでいる。

「将来の拡張に備えて抽象化する」か「今必要な分だけ書く」かは、実務で何度も向き合う判断です。

</details>
