# 第 8 章 例外処理

頻度: ★★★★☆ — 外部 API・ファイル・DB を触るコードには必ず出てきます。

この章で読めるようになるコード:

```python
raise ValueError(f"input_type は document か query: {input_type}")
raise SystemExit(f"chunks に model={args.model} の行がありません。先に load_kb.py を実行してください")
try:
    import anthropic
except ImportError:
    sys.exit("--router には Anthropic SDK が必要です: pip install anthropic")
except urllib.error.HTTPError as e:
    if e.code in (429, 500, 502, 503) and attempt < 3:
        ...
    raise
```

---

## 8.1 トレースバック（エラー表示）の読み方

まず、エラーが起きたときの表示の読み方です。build_kb.py の `meta["page_key"]` が無いページを読んだとすると、
次のように表示されます（簡略化した例）。

```
Traceback (most recent call last):
  File "build_kb.py", line 9, in <module>
    main()
  File "build_kb.py", line 7, in main
    chunk_page(m)
  File "build_kb.py", line 2, in chunk_page
    return {"page_key": meta["page_key"]}
                        ~~~~^^^^^^^^^^^^
KeyError: 'page_key'
```

- **一番下の行から読む**: `KeyError: 'page_key'` が「何が起きたか」（例外の種類: メッセージ）
- その直前が **エラーが起きた場所**（ファイル・行番号・関数名・その行のコード）
- 上に行くほど「誰がそこを呼んだか」をさかのぼる（`most recent call last` = 最後の呼び出しが一番下）

PHP のスタックトレースは「上が最新」のことが多いですが、Python は **下が最新** です。
Python 3.11 以降は `~~~~^^^^` で、行の中のどの部分でエラーが起きたかまで示してくれます。

---

## 8.2 例外を投げる（raise）

### 題材コード

`scripts/embedders.py:46-47`

```python
if input_type not in ("document", "query"):
    raise ValueError(f"input_type は document か query: {input_type}")
```

`scripts/embedders.py:73-74`

```python
if len(vec) != self.dim:
    raise RuntimeError(f"次元が {len(vec)} で返った（期待 {self.dim}）")
```

### 解説

`raise 例外クラス(メッセージ)` で例外を投げます（PHP / JS の `throw new Exception(...)`。`new` は不要）。

**例外の種類（クラス）を意味に合わせて選ぶ** のが実務の作法です。よく使う組み込みの例外は次のとおりです。

| 例外 | いつ使う / 起きる | 題材 |
|---|---|---|
| `ValueError` | 型は合っているが **値がおかしい** 引数 | embedders: `input_type` が想定外 |
| `TypeError` | **型が違う**（`"a" + 1` など） | |
| `KeyError` | 辞書に無いキーを `d[key]` で読んだ | build_kb の `meta["page_key"]` |
| `IndexError` | リストの範囲外を読んだ | |
| `FileNotFoundError` | ファイルが無い | |
| `RuntimeError` | 上のどれにも当てはまらない **実行時の異常** | embedders: API が想定外の次元で返した |
| `NotImplementedError` | 「サブクラスで実装すること」 | ref_embedders（第 10 章） |
| `ImportError` | import できない | eval の任意の依存 |

embedders.py の 2 つの `raise` は性質が違います。

- `ValueError` — **呼び出し側のミス**（間違った引数を渡した）を早く知らせる
- `RuntimeError` — **外部（API）の想定外の振る舞い** を検知して止める。
  256 次元のつもりで 1024 次元が返ってきたのに気付かず DB に入れると、後で原因を追うのが非常に難しくなります。
  「おかしいと分かった時点で止める」= 第 2 章で出てきた **フェイルファスト** です。

メッセージには **何が期待で、実際に何が来たか** を入れるのがコツです（`次元が 1024 で返った（期待 256）`）。

---

## 8.3 スクリプトを終了させる — SystemExit と sys.exit

### 題材コード

`scripts/export_kb.py:45-47`

```python
# 2. 0 件なら load_kb.py の実行忘れなので止める（空の kb.json を Lambda に載せない）
if not rows:
    raise SystemExit(f"chunks に model={args.model} の行がありません。先に load_kb.py を実行してください")
```

`reference/okubo-shamisen/eval.py:82-83`

```python
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("--router には ANTHROPIC_API_KEY が必要です")
```

### 解説

`sys.exit(メッセージ)` と `raise SystemExit(メッセージ)` は **同じ意味** です（`sys.exit` は中で `SystemExit` を投げています）。

- メッセージを標準エラー出力に表示し、**終了コード 1**（失敗）でプログラムを終える
- `sys.exit(0)` は正常終了、`sys.exit(1)` はメッセージなしの異常終了
- トレースバックは表示されない

| 使い分け | 使うもの |
|---|---|
| **利用者の操作ミス・準備不足**（キー未設定、前の手順の実行忘れ） | `sys.exit("どうすればよいか")` — トレースバックは不要、対処法を伝える |
| **プログラムのバグ・想定外の状態** | `raise ValueError(...)` など — トレースバックで場所を特定したい |

export_kb.py のメッセージは「何が起きたか（行が無い）」に加えて「**次に何をすればよいか（load_kb.py を実行）**」まで書いています。
利用者向けのエラーメッセージのお手本です。

終了コードは、シェルスクリプトや CI が成否を判断するのに使います。eval.py の `--strict` は、
誤答が 1 件でもあれば `sys.exit(1)` で CI を落とす仕組みです（eval.py:410-413）。

---

## 8.4 例外を捕まえる（try / except）

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

```python
try:
    失敗するかもしれない処理
except 例外クラス as e:
    その例外が起きたときの処理（e に例外オブジェクトが入る）
```

- **`except` には捕まえる例外の種類を書く**。ここでは `HTTPError`（HTTP のエラー応答）だけを捕まえ、
  ネットワーク切断など別の例外はそのまま上に伝わります。
- `as e` で例外オブジェクトを受け取り、`e.code`（HTTP ステータスコード）を見て判断しています。
- **`raise` だけを書くと、今捕まえた例外をそのまま投げ直します**。
  「429 / 5xx ならリトライ、それ以外（400 や 401 など、何度やっても失敗するもの）は諦めて呼び出し元に任せる」。

> **JS / PHP との比較**
> `try { } catch (HttpError $e) { }` と同じ構造です。JS は `catch (e)` で種類を絞れないので
> `if (e instanceof HttpError)` と書きますが、Python は `except` で種類を指定できます。

### 複数の except・else・finally

```python
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except FileNotFoundError:
    data = {}                           # ファイルが無ければ空で始める
except json.JSONDecodeError as e:
    sys.exit(f"{path} が壊れています: {e}")
else:
    print("読み込み成功")                # 例外が起きなかったときだけ
finally:
    print("必ず実行")                    # 例外が起きても起きなくても
```

| 節 | いつ実行されるか |
|---|---|
| `except A:` | A（またはそのサブクラス）が起きたとき。上から順に照合 |
| `except (A, B):` | A か B が起きたとき |
| `else:` | `try` で例外が起きなかったとき |
| `finally:` | 必ず（後片付け用。ただしファイルや DB の後片付けは `with` を使うのが普通） |

---

## 8.5 任意の依存を ImportError で扱う

### 題材コード

`reference/okubo-shamisen/eval.py:78-81`

```python
try:
    import anthropic
except ImportError:
    sys.exit("--router には Anthropic SDK が必要です: pip install anthropic")
```

### 解説

第 6 章で見たとおり、`--router` を付けたときだけ必要なライブラリを、必要になった時点で import しています。
入っていなければ `ImportError`（正確にはそのサブクラスの `ModuleNotFoundError`）が起きるので、
それを捕まえて **トレースバックの代わりに「何をインストールすればよいか」を表示** しています。

例外を **「別の、利用者に分かりやすい形」に翻訳する** のは、`except` の重要な使い道の 1 つです。

---

## 8.6 わざと捕まえない — フェイルファスト

### 題材コード

`scripts/build_kb.py:65-69`

```python
chunks.append({
    # meta["..."] は、コメント行が欠けたページがあれば KeyError で止まる。
    # 空文字のまま DB の NOT NULL 列に入るのを防ぐため
    "page_key": meta["page_key"],
    "title": meta["title"],
    ...
})
```

### 解説

このコードは **例外を捕まえず、わざと KeyError で止まるようにしています**。

例外処理で一番多い失敗は「とりあえず全部捕まえて握りつぶす」ことです。

```python
# ダメな例
try:
    page_key = meta["page_key"]
except Exception:
    page_key = ""          # エラーが消えるが、空文字のデータが DB に入る
```

こうすると、プログラムは止まりませんが **壊れたデータが静かに広がり**、問題に気付くのがずっと後になります。

例外を捕まえてよいのは、次のどれかに当てはまるときです。

1. **その場で正しく対処できる**（リトライする、既定値で問題ない、別の手段がある）
2. **利用者に分かりやすく伝え直す**（ImportError → インストール方法を案内）
3. **後片付けをしてから投げ直す**（`raise` で再送出）

どれにも当てはまらないなら、捕まえずに上に伝えるのが正解です。

### 絶対に書かないもの

```python
try:
    ...
except:            # 何でも捕まえる（Ctrl+C の KeyboardInterrupt まで）
    pass           # そして何もしない
```

`except:`（種類なし）や `except Exception: pass` は、バグを隠す最悪の書き方です。
どうしても広く捕まえるなら `except Exception as e:` にして、**必ずログに出す** か **投げ直す** こと。

---

## 8.7 LBYL と EAFP — 2 つのスタイル

Python のコミュニティには、エラーへの備え方として 2 つのスタイルがあります。

| スタイル | 意味 | 例 |
|---|---|---|
| **LBYL**（Look Before You Leap: 跳ぶ前に見よ） | 先に確認してから実行 | `if p.exists(): data = json.loads(p.read_text())` |
| **EAFP**（Easier to Ask Forgiveness than Permission: 許可より謝罪） | とりあえず実行し、失敗したら例外で対処 | `try: data = json.loads(p.read_text()) except FileNotFoundError: ...` |

embedders.py のキャッシュ読み込みは `if p.exists():` の **LBYL** です。
Python では **EAFP の方が好まれる** と言われます。確認と実行の間にファイルが消える（競合）可能性がなく、
確認の分だけ処理も減るからです。ただし、単純なスクリプトでは LBYL の方が読みやすいことも多く、
どちらも実務で普通に使われます。「EAFP という考え方がある」と知っておけば十分です。

`d.get(key)` vs `d[key]`、`os.environ.get()` vs `os.environ[]` の使い分けも、この考え方の延長にあります。

---

## 8.8 自分の例外クラスを作る

ライブラリやアプリが大きくなると、独自の例外を定義します。

```python
class EmbeddingError(Exception):
    """埋め込み API に関するエラー。"""


class DimensionMismatch(EmbeddingError):
    def __init__(self, got: int, expected: int):
        super().__init__(f"次元が {got} で返った（期待 {expected}）")
        self.got = got
        self.expected = expected
```

- `Exception` を継承するだけで作れます（クラスと継承は第 10 章）
- 呼び出し側は `except EmbeddingError:` で、この種類のエラーだけをまとめて捕まえられます
- 例外を別の例外に包み直すときは `raise NewError(...) from e` と書くと、元の原因もトレースバックに残ります

---

## 章末問題

### 問 1（選ぶ）次の状況で投げるのに適した例外は？

1. 関数 `make_embedder(provider)` に `"openai"` という未対応の名前が渡された
2. API から返ってきた JSON に、必須のはずの `"embeddings"` キーが無かった
3. 利用者が `--model` に存在しないモデル名を指定して実行した（CLI ツール）

### 問 2（読解）

ref_embedders.py のリトライのループで、サーバが **401 Unauthorized**（API キーが間違っている）を返したとき、
何回リクエストが送られ、最終的に何が起きますか？

### 問 3（書く）

JSON ファイルを読む関数 `load_json(path: Path) -> dict` を書いてください。

- ファイルが無ければ空の辞書を返す
- JSON として壊れていたら `sys.exit` で「ファイル名」と「壊れている旨」を表示して終了する

### 問 4（バグ探し）次のコードの問題点は？

```python
def get_dim(kb: dict) -> int:
    try:
        return kb["dim"]
    except Exception:
        return 256
```

### 問 5（書く）

`retry(func, times=3)` という関数を書いてください。`func()` を呼び、`ConnectionError` が起きたら
1 秒待って再試行し、`times` 回とも失敗したら最後の例外をそのまま投げます。

---

## 回答

<details>
<summary>問 1 の回答</summary>

1. `ValueError` — 呼び出し側が想定外の **値** を渡した（ref_embedders.py の `make_embedder` は `SystemExit` を使っていますが、
   CLI 専用の関数だからです。ライブラリとして使う関数なら `ValueError` が適切）
2. `RuntimeError`（または独自の例外）— 外部が想定外の振る舞いをした。`KeyError` が自然に起きるのに任せてもよいが、
   「API の応答が不正」と分かるメッセージを付けて投げ直す方が親切
3. `sys.exit("...")` — 利用者の操作ミスなので、トレースバックではなく対処法を表示して終了する

</details>

<details>
<summary>問 2 の回答</summary>

**1 回だけ** 送られます。401 は `(429, 500, 502, 503)` に含まれないので、`if` の条件が偽になり、すぐ `raise` で
`HTTPError` が投げ直されます。キーの間違いはリトライしても直らないので、待たずに失敗させるのが正しい動きです。

</details>

<details>
<summary>問 3 の回答</summary>

```python
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        sys.exit(f"{path} は JSON として壊れています: {e}")
```

`json.JSONDecodeError` は `ValueError` のサブクラスなので、`except ValueError` でも捕まえられます。

</details>

<details>
<summary>問 4 の回答</summary>

- `except Exception` が広すぎます。`kb` が辞書でない（`None` が渡された）などの **バグまで 256 で握りつぶし**、気付けなくなります。
- そもそも「無ければ 256」なら例外処理は不要で、`return kb.get("dim", 256)` と書けます。
- さらに言えば、次元は DB・埋め込み・kb.json で一致していないといけない値なので、
  **無いならエラーにすべき**（`kb["dim"]` のまま）かもしれません。既定値を入れてよいかは、値の意味から判断します。

</details>

<details>
<summary>問 5 の回答</summary>

```python
import time


def retry(func, times=3):
    for attempt in range(times):
        try:
            return func()
        except ConnectionError:
            if attempt == times - 1:
                raise          # 最後の 1 回は投げ直す
            time.sleep(1)
```

ref_embedders.py のリトライと同じ構造です。`return func()` で成功したらその場で値を返すので、`break` は不要です。

</details>
