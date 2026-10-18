# 第 6 章 モジュールと import・`if __name__ == "__main__"`

頻度: ★★★★☆ — すべてのファイルの冒頭と末尾にあります。

この章で読めるようになるコード:

```python
import json
from pathlib import Path
import numpy as np
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedders import CHUNKS_PATH, DEFAULT_MODEL, Embedder, chunk_to_text  # noqa: E402

if __name__ == "__main__":
    main()
```

---

## 6.1 モジュール = 1 つの .py ファイル

Python では **1 つの `.py` ファイルが 1 つの「モジュール」** です。
`scripts/embedders.py` は `embedders` というモジュールで、ほかのファイルから `import embedders` で使えます。

- モジュールのトップレベル（関数やクラスの外）に書いた変数・関数・クラスは、すべて外から import できます。
- PHP の `require` / `include` が「ファイルの中身をその場に貼り付ける」のに対し、
  Python の import は **「そのファイルを 1 回だけ実行し、できあがった名前の入れ物を受け取る」** 仕組みです。
  同じモジュールを何度 import しても、実行されるのは最初の 1 回だけです。
- JS の ES Modules（`import { x } from "./x.js"`）に近い感覚ですが、Python は **`export` を書かなくても全部公開** されます。
  「外から使わないで」という意思表示は、名前の先頭に `_` を付けることで行います（`_STRIP`、`_CachedEmbedder`）。

---

## 6.2 import の書き方

### 題材コード

`scripts/load_kb.py:12-24`

```python
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

# python -I（Dockerfile の ENTRYPOINT）はスクリプトのフォルダを import 先に入れないため、明示的に足す
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedders import CHUNKS_PATH, DEFAULT_MODEL, Embedder, chunk_to_text  # noqa: E402
```

### 解説

| 書き方 | 使い方 | 意味 |
|---|---|---|
| `import json` | `json.loads(...)` | モジュール全体を `json` という名前で使う |
| `from pathlib import Path` | `Path(...)` | モジュールの中の `Path` だけを直接使う |
| `import numpy as np` | `np.array(...)` | 別名を付ける（長い名前の短縮。`np` は世界共通の慣習） |
| `from pgvector.psycopg import register_vector` | `register_vector(conn)` | パッケージ（フォルダ）の中のモジュールから取り出す |
| `from embedders import A, B, C` | `Embedder(...)` | 複数まとめて |

`import json` と `from json import loads` のどちらを使うかは好みもありますが、
**モジュール名を残す方（`json.loads`）がどこから来た関数か分かりやすい** ので、標準ライブラリは `import xxx` が多めです。
クラス（`Path`、`Decimal`）は `from ... import` で直接取り込むのが一般的です。

`from x import *`（全部取り込む）は、どこから来た名前か分からなくなるので **実務では使いません**。

### import の並べ順（PEP 8 の慣習）

load_kb.py の import は空行で 3 つのグループに分かれています。

1. **標準ライブラリ**（Python に最初から入っている）: `argparse`, `json`, `os`, `sys`, `pathlib`
2. **サードパーティ**（pip で入れた）: `numpy`, `psycopg`, `pgvector`
3. **自分のプロジェクトのモジュール**: `embedders`

各グループの中はアルファベット順。この並びは `ruff` や `isort` といったツールで自動整形できます（第 15 章）。

> **PHP との比較**
> Laravel の `use App\Models\User;` と Composer のオートロードに慣れていると、
> 「ファイルの場所 = 名前空間」という点は似ています。ただし Python は **フォルダ構成そのものがパッケージ名** になり、
> `composer.json` の `autoload` に当たる設定は通常いりません。

---

## 6.3 sys.path と `# noqa: E402`

### 解説

`import embedders` と書いたとき、Python は **`sys.path` というリストに入っているフォルダを順に探します**。
通常は「実行したスクリプトのあるフォルダ」が先頭に自動で入るので、同じフォルダの `embedders.py` が見つかります。

ところが、このリポジトリの Docker は `python -I`（isolated mode）で実行しています（`docker/python/Dockerfile`）。

```dockerfile
ENTRYPOINT ["python", "-I"]
```

`-I` は「環境変数やカレントフォルダの影響を受けない、安全な実行モード」で、
**スクリプトのフォルダも `sys.path` に入れません**。そのため `import embedders` が失敗します。
そこで、

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
```

で、このファイルのあるフォルダ（`scripts/`）を探索先の先頭に手で足しています。

- `__file__` — 今実行しているファイルのパス（PHP の `__FILE__` と同じ）
- `.resolve()` — 絶対パスにする（シンボリックリンクも解決）
- `.parent` — 親フォルダ（PHP の `__DIR__` に当たるものが `Path(__file__).resolve().parent`）
- `str(...)` — `sys.path` には文字列で入れる

`# noqa: E402` は **リンター（コードの検査ツール）への指示** です。
E402 は「import はファイルの先頭にまとめて書け」という規則で、`sys.path.insert` の後に import を書いているため
違反になります。「わざとなので無視して」と書いておくのが `# noqa`（no quality assurance）です。

> 実務の大きめのプロジェクトでは、`sys.path` をいじる代わりに **パッケージとしてインストール**（`pip install -e .`）して
> import できるようにするのが一般的です。小さなスクリプト集なら、この書き方で十分です。

---

## 6.4 パッケージ（フォルダ）とサードパーティ

`from pgvector.psycopg import register_vector` の `pgvector.psycopg` は、
`pgvector` という **パッケージ（フォルダ）** の中の `psycopg` という **モジュール（ファイル）** を指しています。

```
pgvector/
├── __init__.py      ← これがあるとフォルダがパッケージになる（中身は空でもよい）
├── psycopg.py       ← pgvector.psycopg
└── sqlalchemy.py
```

サードパーティのパッケージは **pip** でインストールします（PHP の Composer、JS の npm に当たる）。
このリポジトリでは `docker/python/requirements.txt` に一覧があり、Docker イメージを作るときに入ります。

```
voyageai>=0.3
psycopg[binary]>=3.1
pgvector>=0.3
numpy>=1.26
```

| | Python | PHP | JS |
|---|---|---|---|
| パッケージ管理 | pip（+ venv / uv / poetry） | Composer | npm / pnpm |
| 依存の一覧 | `requirements.txt` / `pyproject.toml` | `composer.json` | `package.json` |
| インストール先 | 仮想環境（venv）やコンテナ | `vendor/` | `node_modules/` |

Python は **プロジェクトごとに仮想環境（venv）を作る** のが基本です（第 15 章）。このリポジトリは代わりに
Docker コンテナを使ってホストを汚さないようにしています。

---

## 6.5 import を遅らせる（任意の依存）

### 題材コード

`reference/okubo-shamisen/eval.py:77-84`

```python
def __init__(self, model: str, school_name: str):
    try:
        import anthropic
    except ImportError:
        sys.exit("--router には Anthropic SDK が必要です: pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("--router には ANTHROPIC_API_KEY が必要です")
    self.client = anthropic.Anthropic()
```

### 解説

`import` はファイルの先頭でなくても、**関数の中でも書けます**。

- eval.py は `--router` オプションを付けたときだけ Anthropic の API を使います。
- 先頭で `import anthropic` すると、**使わないときでも SDK が入っていないとエラーで起動できません**。
- 関数の中で import し、`ImportError` を捕まえて「入れてください」と案内することで、
  「普段は無くても動き、必要なときだけ分かりやすいエラーを出す」ようにしています。

これは「オプション機能の依存を必須にしない」ための実務的なテクニックです。例外処理（`try` / `except`）は第 8 章で扱います。

---

## 6.6 環境変数 os.environ

### 題材コード

`scripts/load_kb.py:45` / `reference/okubo-shamisen/eval.py:82`

```python
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:     # 無ければ KeyError で止まる
...
if not os.environ.get("ANTHROPIC_API_KEY"):                  # 無ければ None
```

### 解説

`os.environ` は環境変数が入った **辞書のようなもの** です。第 2 章の `d[key]` と `d.get(key)` の使い分けがそのまま当てはまります。

- DB の接続先は必須 → `os.environ["DATABASE_URL"]`（無ければ即エラー）
- API キーは `--router` のときだけ必要 → `.get()` で確認し、分かりやすいメッセージで終了

API キーなどの秘密情報は **コードに書かず、環境変数で渡す** のが鉄則です（PHP の `getenv()` / Laravel の `env()`、
Node の `process.env` と同じ）。このリポジトリでは `.env` を `make py` が読み込んでコンテナに渡しています。

---

## 6.7 `if __name__ == "__main__":`

### 題材コード

`scripts/build_kb.py:110-111`（ほぼすべての題材ファイルの末尾）

```python
if __name__ == "__main__":
    main()
```

### 解説

Python のファイルは 2 通りの使われ方をします。

1. **スクリプトとして直接実行される**: `python scripts/embedders.py`
2. **モジュールとして import される**: load_kb.py の `from embedders import Embedder`

`__name__` は Python が自動で用意する変数で、

- 直接実行されたときは `"__main__"`
- import されたときはモジュール名（`"embedders"`）

になります。つまりこの `if` は **「直接実行されたときだけ `main()` を動かす」** という意味です。

embedders.py は、単体で実行すると「全チャンクを埋め込んで件数を表示する」ツールとして動き、
load_kb.py から import したときは `Embedder` クラスなどの **部品だけを提供** します。
この `if` が無いと、load_kb.py が `import embedders` した瞬間に embedders.py の `main()` まで走ってしまいます。

> **JS / PHP との比較**
> JS（Node）の `if (require.main === module)` や、ES Modules の `import.meta.main`（Deno / Bun）に当たります。
> PHP にはこれに当たる標準的な書き方はなく、「ライブラリ用ファイル」と「実行用ファイル」を分けるのが普通です。

### main() 関数にまとめる理由

`if __name__ == "__main__":` の下に処理を直接書くこともできますが、題材はすべて `main()` 関数にまとめています。

- `main()` の中の変数はローカル変数になり、モジュール全体に漏れない
- テストやほかのスクリプトから `main()` を呼べる
- 処理の入口がひと目で分かる

実務でも「`def main()` + `if __name__ == "__main__": main()`」が標準の形です。

---

## 6.8 モジュールのトップレベルに置くもの

題材のファイルは、どれもおおむね次の順番で並んでいます。

```python
"""ファイルの説明（docstring）"""          # 1. 何をするファイルか・使い方

import ...                                   # 2. import

ROOT = Path(__file__).resolve().parent.parent   # 3. 定数（パス・設定値）
CHUNK_MAX = 450

def parse_page(...): ...                     # 4. 部品の関数・クラス
def chunk_page(...): ...

def main() -> None: ...                      # 5. 入口

if __name__ == "__main__":                   # 6. 直接実行されたときだけ
    main()
```

ファイル冒頭の docstring に **「使い方」のコマンド例** まで書いてあるのが、このリポジトリの良い習慣です。
1 か月後の自分やチームメンバーが、コードを読まずに使い始められます。

---

## 章末問題

### 問 1（読解）

`scripts/export_kb.py` は `from embedders import DEFAULT_MODEL, DIM` で定数を取り込んでいます。
`DIM = 256` を export_kb.py にもう一度書かずに import している理由を考えてください。

### 問 2（読解）

次の 2 ファイルがあるとき、`python b.py` を実行すると何が表示されますか？

```python
# a.py
print("a が読み込まれた")
def hello():
    print("hello")
if __name__ == "__main__":
    print("a を直接実行した")
```

```python
# b.py
import a
import a
a.hello()
print(__name__)
```

### 問 3（書く）

ファイル `greet.py` を作り、次の条件を満たしてください。

- `greet(name: str) -> str` 関数（`f"こんにちは、{name}さん"` を返す）
- 直接実行したときだけ `greet("三味線")` の結果を表示する
- 別のファイルから `from greet import greet` しても何も表示されない

### 問 4（考える）

`from embedders import *` と書かない方がよい理由を 2 つ挙げてください。

### 問 5（調べる）

REPL で `import sys; print(sys.path)` を実行し、何が入っているか確かめてください。
次に、`python3 -I -c "import sys; print(sys.path)"` と比べ、先頭の要素の違いを確認してください。

---

## 回答

<details>
<summary>問 1 の回答</summary>

`DIM` は「DB の `vector(256)`・埋め込み・kb.json」で **必ず一致していなければならない値** だからです。
複数のファイルに `256` を書くと、片方だけ変えたときに不整合が起きます。
**値の出どころを 1 か所にする（Single Source of Truth）** ため、embedders.py に定義して他は import しています。
embedders.py の `DIM` のコメントに「`db/init/001_schema.sql` の `vector(256)` と一致させる」とあるのも同じ意図です。

</details>

<details>
<summary>問 2 の回答</summary>

```
a が読み込まれた
hello
__main__
```

- `import a` を 2 回書いても、モジュールが実行されるのは **最初の 1 回だけ** です。
- `a.py` は import されたので `__name__` は `"a"` になり、`"a を直接実行した"` は表示されません。
- `b.py` は直接実行されたので、`b.py` の中の `__name__` は `"__main__"` です。

</details>

<details>
<summary>問 3 の回答</summary>

```python
"""挨拶文を作る。

使い方: python3 greet.py
"""


def greet(name: str) -> str:
    return f"こんにちは、{name}さん"


def main() -> None:
    print(greet("三味線"))


if __name__ == "__main__":
    main()
```

</details>

<details>
<summary>問 4 の回答</summary>

1. **どこから来た名前か分からなくなる**: `Embedder` を見ても、どのファイルを見ればよいか読み手が探せない。
2. **名前の衝突が見えない**: 取り込んだ名前が、自分のファイルの変数や別の import を黙って上書きすることがある
   （embedders.py にも `main` や `ROOT` があり、取り込んだ側の同名のものとぶつかる）。

加えて、リンターが未使用の import や未定義の名前を検出できなくなる、という実務上の不利もあります。

</details>

<details>
<summary>問 5 の回答</summary>

通常の実行では、先頭に `''`（カレントフォルダ）やスクリプトのフォルダが入ります。
`-I` を付けると、それが入りません。そのため `-I` で実行するこのリポジトリのスクリプトは、
`sys.path.insert(0, ...)` で自分のフォルダを足しています。

</details>
