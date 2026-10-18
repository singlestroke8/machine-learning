# 第 13 章 argparse でコマンドラインツールを作る

頻度: ★★★☆☆ — 実務の Python は「コマンドで動かす小さなツール」が大量にあります。バッチ・データ変換・評価スクリプトの入口は、ほぼこの形です。

この章で読めるようになるコード:

```python
parser = argparse.ArgumentParser(description="販売管理 DataSet のダミー CSV を生成する")
parser.add_argument("--orders", type=int, default=23000, help="案件数(...)")
parser.add_argument("--out-dir", type=Path, default=Path("dummy"), help="...")
parser.add_argument("--encoding", choices=["utf-8", "cp932"], default="utf-8", help="...")
ap.add_argument("--router", action="store_true", help="...")
args = parser.parse_args()
```

---

## 13.1 なぜ argparse か

コマンドラインの引数は `sys.argv`（文字列のリスト）でも読めます。

```bash
python3 scripts/embedders.py --model voyage-3.5
```

```python
import sys
print(sys.argv)   # ['scripts/embedders.py', '--model', 'voyage-3.5']
```

しかし自分で解析すると、「`--model` の次の要素を取る」「数値に変換する」「間違った値ならエラーを出す」「`--help` を表示する」
をすべて手で書くことになります。標準ライブラリの **argparse** は、これを宣言的に書くだけで全部やってくれます。

> **PHP / JS との比較**
> PHP の `getopt()`、Laravel の Artisan コマンドの `$signature = 'app:import {--model=}'`、
> Node の `commander` / `yargs` に当たります。Python は標準ライブラリで十分な機能が揃っています。

---

## 13.2 最小の形 — scripts/*.py

### 題材コード

`scripts/embedders.py:90-93`

```python
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
```

### 解説

1. `ArgumentParser()` で解析器を作る
2. `add_argument("--model", default=...)` で「`--model` というオプションがあり、無ければ既定値」と宣言する
3. `parse_args()` で実際のコマンドラインを解析し、結果を **`args` オブジェクト** にまとめる
4. `args.model` で値を読む

```bash
python3 scripts/embedders.py                       # args.model == "voyage-3.5-lite"
python3 scripts/embedders.py --model voyage-3.5    # args.model == "voyage-3.5"
python3 scripts/embedders.py --model=voyage-3.5    # = でつないでもよい
```

scripts/ の 4 本はすべて同じ `--model` を持っていて、**Phase 3 の Embedding 比較で同じ手順をモデルを変えて流せる** ようになっています。

---

## 13.3 型・選択肢・ヘルプ — dummy_csv

### 題材コード

`PivotAggregationSystem/scripts/generate_dummy_csv.py:497-505`

```python
parser = argparse.ArgumentParser(description="販売管理 DataSet のダミー CSV を生成する")
parser.add_argument("--orders", type=int, default=23000, help="案件数(既定 23000 = 実データと同程度。明細は約 53,000 行)")
parser.add_argument("--out-dir", type=Path, default=Path("dummy"), help="出力先のフォルダ(既定 dummy。.gitignore 済み)")
# 既定は元システムのエクスポートと同じ名前(取り込みでは「【業務】販売管理DB：」で始まる名前だけを受け付ける)
parser.add_argument("--name", default="【業務】販売管理DB：管理者用", help="ファイル名の元(既定 【業務】販売管理DB：管理者用)")
parser.add_argument("--seed", type=int, default=42, help="乱数の種。同じ値なら同じ CSV になる(既定 42)")
parser.add_argument("--encoding", choices=["utf-8", "cp932"], default="utf-8", help="文字コード(既定 utf-8)")
parser.add_argument("--split", type=int, default=6, help="分割するファイル数(既定 6。元システムの分割ダウンロードを再現する。1 なら分割しない)")
args = parser.parse_args()
```

### 解説

| 引数 | 意味 | 例 |
|---|---|---|
| `type=int` | 文字列を **int に変換** してから渡す。変換できなければエラー | `--orders 100` → `args.orders == 100`（数値） |
| `type=Path` | `Path("...")` に変換。**型として使えるのは「文字列を 1 つ受け取って値を返す関数（クラス）」なら何でも** | `--out-dir tmp/` → `Path("tmp")` |
| `choices=[...]` | この中のどれかしか受け付けない | `--encoding sjis` → エラー |
| `default=...` | 指定されなかったときの値 | |
| `help="..."` | `--help` で表示される説明 | |
| `description=...` | ツール全体の説明 | |

**`--out-dir` は `args.out_dir` で読みます**。オプション名の `-` は、属性名では `_` に自動で変わります（Python の名前に `-` は使えないため）。

`type=Path` にしているので、`main()` の中でいきなり `args.out_dir.mkdir(parents=True, exist_ok=True)` や
`args.out_dir / f"{args.name}.csv"` と Path のメソッドが使えます。**入口で型を揃えておくと、中の処理が簡単になります**。

### 自動で付いてくるもの

`--help`（`-h`）は書かなくても自動で付き、宣言からヘルプを作ってくれます。

```
$ python3 scripts/generate_dummy_csv.py --help
usage: generate_dummy_csv.py [-h] [--orders ORDERS] [--out-dir OUT_DIR]
                             [--name NAME] [--seed SEED]
                             [--encoding {utf-8,cp932}] [--split SPLIT]

販売管理 DataSet のダミー CSV を生成する

options:
  -h, --help            show this help message and exit
  --orders ORDERS       案件数(既定 23000 = 実データと同程度。明細は約 53,000 行)
  --out-dir OUT_DIR     出力先のフォルダ(既定 dummy。.gitignore 済み)
  ...
```

間違った値を渡すと、使い方と分かりやすいエラーを表示して **終了コード 2** で止まります。

```
$ python3 scripts/generate_dummy_csv.py --split abc
generate_dummy_csv.py: error: argument --split: invalid int value: 'abc'

$ python3 scripts/generate_dummy_csv.py --encoding sjis
generate_dummy_csv.py: error: argument --encoding: invalid choice: 'sjis' (choose from 'utf-8', 'cp932')
```

自分で `if args.encoding not in (...)` を書かなくても、**宣言しただけで入力チェックが付く** のが argparse の強みです。

### help に既定値と理由を書く

dummy_csv の `help` には「既定 23000 = 実データと同程度」「.gitignore 済み」「同じ値なら同じ CSV になる」のように、
**既定値とその理由** まで書かれています。ヘルプはそのまま使い方のドキュメントになるので、ここに情報を集めるのは良い習慣です。

---

## 13.4 フラグ（on / off）— action="store_true"

### 題材コード

`reference/okubo-shamisen/eval.py:208-216`

```python
ap = argparse.ArgumentParser()
ap.add_argument("--provider", choices=["voyage", "mock"], default="mock")
ap.add_argument("--model", default=None)
ap.add_argument("--kb", default=str(RAG_DIR / "dist/kb.json"))
ap.add_argument("--router", action="store_true",
                help="判定段（Tier1r）を実際にAnthropic APIで評価する（要 ANTHROPIC_API_KEY・anthropic SDK）")
ap.add_argument("--strict", action="store_true",
                help="想定問答の質問文（＋軽い変形）が1件でも定型回答できなければ終了コード1")
args = ap.parse_args()
```

### 解説

`action="store_true"` は **値を取らないオプション**（フラグ）です。付ければ `True`、付けなければ `False`。

```bash
python3 eval.py                     # args.router == False, args.strict == False
python3 eval.py --router --strict   # 両方 True
```

使う側では、

```python
router = Router(cfg["router_model"], cfg["school_name"]) if args.router else None
...
if args.strict and (pipe_fails or neg_canned or pos_wrong):
    sys.exit(1)
```

- `--router` を付けたときだけ有料の API を呼ぶ（付けなければ無料の範囲で評価）
- `--strict` を付けると、取りこぼしが 1 件でもあれば終了コード 1 で終わる → **CI で「精度が落ちたらビルドを落とす」** に使える

**お金がかかる・副作用がある動作は、既定では off にしてフラグで明示的に on にする** のが安全な設計です。
`--model` の既定値が `None` になっているのは「指定が無ければ kb.json に記録されたモデルを使う」ためで、
`args.model or kb["model"]` と組み合わせています（第 1 章の `or`）。

---

## 13.5 位置引数・短いオプション・その他

題材には出てきませんが、よく使う書き方です。

```python
ap = argparse.ArgumentParser(description="CSV の行数を数える")
ap.add_argument("paths", nargs="+", type=Path, help="対象のファイル（複数可）")  # 位置引数（- が付かない）
ap.add_argument("-n", "--limit", type=int, default=10)                            # 短い名前と長い名前
ap.add_argument("-v", "--verbose", action="store_true")
ap.add_argument("--dry-run", action="store_true", help="書き込まずに結果だけ表示")
ap.add_argument("--model", required=True)                                         # 必須のオプション
args = ap.parse_args()
```

| 書き方 | 意味 |
|---|---|
| `"paths"`（`-` なし） | **位置引数**。順番で受け取る。既定では必須 |
| `nargs="+"` / `nargs="*"` | 1 個以上 / 0 個以上をリストで受け取る |
| `"-n", "--limit"` | 短い名前と長い名前の両方 |
| `required=True` | オプションを必須にする |
| `action="count"` | `-vvv` の回数を数える |
| `ap.error("メッセージ")` | 自分で入力チェックしてエラーを出す（argparse と同じ形式で終了） |

`--dry-run`（実際には書き込まず、何が起きるかだけ表示する）は、**削除や DB 更新を伴うツールにはぜひ付けたい** オプションです。

### サブコマンド（参考）

`git commit` / `git push` のような **サブコマンド** は `ap.add_subparsers()` で作れます。
ツールが大きくなってきたら、`click` や `typer` といったサードパーティのライブラリを使うこともあります。

---

## 13.6 テストしやすい main

`parse_args()` は既定で `sys.argv` を読みますが、**リストを渡すとそれを解析** します。

```python
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="voyage-3.5-lite")
    args = ap.parse_args(argv)          # None なら sys.argv を使う
    ...

main(["--model", "voyage-3.5"])        # テストからはこう呼べる
```

題材の `main()` は引数なしですが、テストを書く段階になったらこの形に変えると便利です（第 15 章）。

---

## 章末問題

### 問 1（読解）

dummy_csv を次のように実行したとき、`args` の各値（`orders`, `out_dir`, `seed`, `encoding`, `split`）は？ 型も答えてください。

```bash
python3 scripts/generate_dummy_csv.py --orders 100 --split 1 --out-dir tmp/
```

### 問 2（読解）

次の 2 つの実行は、それぞれどうなりますか？

```bash
python3 eval.py --provider openai
python3 eval.py --router=yes
```

### 問 3（書く）

次の仕様のコマンドラインツール `chunk_stats.py` の `main()` を書いてください。

- 位置引数 `path`: 読み込む JSON ファイル（`.cache/chunks.json` と同じ形: `[{"id": ..., "text": ...}, ...]`）。`Path` で受け取る
- `--min-len`（int、既定 60）: これより短いチャンクの数を数える
- `--verbose`: 付けたら、短いチャンクの id を 1 行ずつ表示する
- 出力: `chunks: 総数 / short(<60): 個数`

### 問 4（書く）

問 3 のツールに、`--min-len` が 0 以下のときに argparse 形式のエラーで終了する処理を足してください。

### 問 5（考える）

load_kb.py は実行すると「同じモデルの行を全部消して入れ直す」ツールです。
このツールに `--dry-run` を足すとしたら、何を表示し、何をしないようにしますか？

---

## 回答

<details>
<summary>問 1 の回答</summary>

| 属性 | 値 | 型 |
|---|---|---|
| `args.orders` | `100` | int |
| `args.out_dir` | `PosixPath('tmp')` | Path（末尾の `/` は消える） |
| `args.seed` | `42`（既定値） | int |
| `args.encoding` | `'utf-8'`（既定値） | str |
| `args.split` | `1` | int |

`--split 1` なので分割せず、`tmp/【業務】販売管理DB：管理者用.csv` の 1 ファイルが出力されます。

</details>

<details>
<summary>問 2 の回答</summary>

1. `error: argument --provider: invalid choice: 'openai' (choose from 'voyage', 'mock')` と表示されて終了（終了コード 2）
2. `error: argument --router: ignored explicit argument 'yes'` と表示されて終了。`store_true` は値を取らないので、`--router` とだけ書きます

</details>

<details>
<summary>問 3 の回答</summary>

```python
"""チャンクの JSON を読み、短いチャンクの数を表示する。

使い方: python3 chunk_stats.py .cache/chunks.json --min-len 80 --verbose
"""

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="チャンクの長さを集計する")
    ap.add_argument("path", type=Path, help="chunks.json のパス")
    ap.add_argument("--min-len", type=int, default=60, help="これ未満を短いとみなす(既定 60)")
    ap.add_argument("--verbose", action="store_true", help="短いチャンクの id を表示する")
    args = ap.parse_args(argv)

    chunks = json.loads(args.path.read_text(encoding="utf-8"))
    short = [c for c in chunks if len(c["text"]) < args.min_len]
    print(f"chunks: {len(chunks)} / short(<{args.min_len}): {len(short)}")
    if args.verbose:
        for c in short:
            print(f"  {c['id']}")


if __name__ == "__main__":
    main()
```

</details>

<details>
<summary>問 4 の回答</summary>

```python
    args = ap.parse_args(argv)
    if args.min_len <= 0:
        ap.error("--min-len は 1 以上を指定してください")
```

`ap.error()` は使い方の表示 + エラーメッセージ + 終了コード 2 で終了します。
`type=` に「正の整数だけを受け付ける関数」を自作して渡す方法もあります。

```python
def positive_int(s: str) -> int:
    n = int(s)
    if n <= 0:
        raise argparse.ArgumentTypeError("1 以上を指定してください")
    return n

ap.add_argument("--min-len", type=positive_int, default=60)
```

</details>

<details>
<summary>問 5 の回答例</summary>

- **表示するもの**: 対象モデル、現在 DB にあるそのモデルの行数（消える予定の件数）、投入予定の件数、埋め込みのキャッシュヒット数と課金が発生しそうなテキスト数
- **しないこと**: `DELETE` / `INSERT` の実行、そして **Voyage API の呼び出し**（課金が発生するため）。
  キャッシュに無いテキストの件数だけを数えて表示する

「DB を変えない」だけでなく「お金がかかる外部呼び出しもしない」までが dry-run の役目です。

</details>
