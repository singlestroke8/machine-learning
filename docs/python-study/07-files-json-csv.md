# 第 7 章 ファイル入出力（pathlib / json / csv）

頻度: ★★★★☆ — スクリプト・バッチ・データ処理では毎日書きます。

この章で読めるようになるコード:

```python
ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / "corpus/pages"
for path in sorted(PAGES_DIR.glob("*.md")):
OUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
with target.open("w", encoding=args.encoding, errors="replace", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
```

---

## 7.1 pathlib — パスをオブジェクトとして扱う

### 題材コード

`scripts/build_kb.py:14-16`

```python
ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / "corpus/pages"
OUT_PATH = ROOT / ".cache/chunks.json"
```

### 解説

`pathlib.Path` はファイルパスを表すクラスです。文字列ではなく **オブジェクト** として扱うので、
パスの結合や名前の取り出しがメソッドでできます。

- `Path(__file__)` — このファイル（`scripts/build_kb.py`）のパス
- `.resolve()` — 絶対パスに
- `.parent.parent` — 親の親 = リポジトリのルート（`scripts/` の 1 つ上）
- **`/` 演算子でパスをつなげる** — `ROOT / "corpus/pages"`。OS に合わせた区切り文字（Windows なら `\`）で結合してくれます

**「スクリプトの場所を基準にパスを組み立てる」** ので、どのフォルダから実行しても同じファイルを指します。
`open("corpus/pages/x.md")` のような相対パスは「実行したときのカレントフォルダ」基準になり、
別の場所から実行すると見つからない、という事故が起きます。

> **PHP との比較**
> `Path(__file__).resolve().parent` は PHP の `__DIR__`、`ROOT / "corpus"` は `__DIR__ . '/../corpus'` に当たります。
> Laravel の `base_path('corpus/pages')` のようなヘルパーを、標準ライブラリだけで作っているイメージです。

### Path の主な属性・メソッド

| 書き方 | 結果（`p = Path("/repo/corpus/pages/price.md")`） | 題材 |
|---|---|---|
| `p.name` | `'price.md'` | dummy_csv `old.name` |
| `p.stem` | `'price'`（拡張子なし） | |
| `p.suffix` | `'.md'` | |
| `p.parent` | `Path('/repo/corpus/pages')` | 多数 |
| `p.exists()` | 存在するか | embedders |
| `p.read_text(encoding="utf-8")` | ファイル全体を文字列で読む | build_kb |
| `p.write_text(s, encoding="utf-8")` | 文字列を書く（上書き） | build_kb, export_kb |
| `p.mkdir(parents=True, exist_ok=True)` | フォルダを作る | embedders, dummy_csv |
| `d.glob("*.md")` | パターンに合うファイルの一覧 | build_kb |
| `d.iterdir()` | フォルダ直下の全エントリ | dummy_csv |
| `p.unlink()` | ファイルを削除 | dummy_csv |
| `p.relative_to(ROOT)` | ROOT からの相対パス | build_kb, export_kb |
| `p.stat().st_size` | ファイルサイズ（バイト） | export_kb |
| `p.open("w", ...)` | ファイルを開く（7.4 で） | dummy_csv |

古いコードでは `os.path.join()` や `os.path.exists()` を使っていますが、**新しく書くなら pathlib** が標準です。

---

## 7.2 テキストファイルを丸ごと読み書きする

### 題材コード

`scripts/build_kb.py:24`

```python
text = path.read_text(encoding="utf-8")
```

`scripts/build_kb.py:100-101`

```python
OUT_PATH.parent.mkdir(exist_ok=True)
OUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
```

### 解説

- `read_text()` / `write_text()` は **開く・読む（書く）・閉じる** を 1 回でやってくれます。
  数 MB 程度までのファイルなら、これが一番簡単で安全です（PHP の `file_get_contents` / `file_put_contents`）。
- **`encoding="utf-8"` は必ず書く**。省略すると OS の既定の文字コードが使われ、
  Windows では cp932（Shift_JIS の一種）になって日本語が化けたりエラーになったりします。
- `OUT_PATH.parent.mkdir(exist_ok=True)` — 書き出し先のフォルダ（`.cache/`）が無ければ作る。
  `exist_ok=True` は「もうあってもエラーにしない」。

### mkdir の parents

`scripts/embedders.py:76`

```python
p.parent.mkdir(parents=True, exist_ok=True)
```

キャッシュのパスは `.cache/embeddings/voyage-3.5-lite-256/<ハッシュ>.json` と **深い階層** です。
`parents=True` は「途中のフォルダも全部作る」（シェルの `mkdir -p`、PHP の `mkdir($dir, 0777, true)`）。

---

## 7.3 JSON の読み書き

### 題材コード

`scripts/embedders.py:55, 77`

```python
out[i] = json.loads(p.read_text())
...
p.write_text(json.dumps(vec))
```

`scripts/export_kb.py:73`

```python
OUT_PATH.write_text(json.dumps(kb, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
```

### 解説

| 関数 | 意味 | JS | PHP |
|---|---|---|---|
| `json.loads(文字列)` | JSON 文字列 → Python の値 | `JSON.parse` | `json_decode($s, true)` |
| `json.dumps(値)` | Python の値 → JSON 文字列 | `JSON.stringify` | `json_encode` |
| `json.load(ファイル)` | 開いたファイルから読む | | |
| `json.dump(値, ファイル)` | 開いたファイルへ書く | | |

`s` の付いている方が **s**tring（文字列）版です。題材は `read_text()` / `write_text()` と組み合わせて文字列版を使っています。

JSON と Python の型は次のように対応します。

| JSON | Python |
|---|---|
| object `{}` | `dict` |
| array `[]` | `list` |
| string | `str` |
| number | `int` / `float` |
| `true` / `false` | `True` / `False` |
| `null` | `None` |

### dumps のオプション

```python
>>> json.dumps({"a": "三味線"})
'{"a": "\\u4e09\\u5473\\u7dda"}'                # 既定は日本語を \u エスケープ
>>> json.dumps({"a": "三味線"}, ensure_ascii=False)
'{"a": "三味線"}'                                # 日本語のまま
>>> json.dumps({"a": 1, "b": [1, 2]}, separators=(",", ":"))
'{"a":1,"b":[1,2]}'                              # 空白なしで最小サイズ
```

| オプション | 意味 | 題材での使い分け |
|---|---|---|
| `ensure_ascii=False` | 日本語をそのまま出す（PHP の `JSON_UNESCAPED_UNICODE`） | 人が読む / サイズを減らす |
| `indent=1` | 改行とインデントで整形（PHP の `JSON_PRETTY_PRINT`） | build_kb: 人が中身を確認する中間ファイル |
| `separators=(",", ":")` | 区切りの後ろの空白をなくす | export_kb: Lambda に載せる本番ファイルは小さく |

**誰が読むファイルか** で書き方を変えている点に注目してください。
embedders.py のキャッシュ（ベクトル = 数値だけ）は日本語を含まないので、オプションなしです。

---

## 7.4 open と with — ファイルを開いて少しずつ書く

### 題材コード

`PivotAggregationSystem/scripts/generate_dummy_csv.py:532-538`

```python
for target in targets:
    # newline="" と lineterminator="\r\n" で、元システムと同じ CRLF 改行にする(値の中の改行は LF のまま)
    # errors="replace": cp932 を指定したとき、表せない文字(♡ など)を「?」にする
    with target.open("w", encoding=args.encoding, errors="replace", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        writer.writerow(HEADERS)  # 分割ファイルにもそれぞれヘッダ行が入る
        writer.writerows(islice(rows_iter, per_file))
```

### 解説

大きなデータを書くときや、CSV のように 1 行ずつ書くときは、ファイルを **開いたまま** 書き込みます。

- `target.open("w", ...)` — 書き込みモードで開く（組み込みの `open(target, "w", ...)` と同じ）
- **`with ... as f:`** — ブロックを抜けるときに **必ずファイルを閉じる**。途中で例外が起きても閉じます（第 12 章で詳しく）

| モード | 意味 |
|---|---|
| `"r"` | 読む（既定） |
| `"w"` | 書く（**既存の中身は消える**） |
| `"a"` | 末尾に追記 |
| `"rb"` / `"wb"` | バイナリで読む / 書く（画像など） |

| 引数 | 意味 |
|---|---|
| `encoding="utf-8"` | 文字コード。dummy_csv は `--encoding cp932` で Excel 向けの Shift_JIS も出せる |
| `errors="replace"` | その文字コードで表せない文字を `?` に置き換える（既定はエラーで停止） |
| `newline=""` | 改行コードの自動変換をしない（CSV では必須。下で説明） |

> **PHP との比較**
> PHP の `$f = fopen(...); ... fclose($f);` の `fclose` を、`with` が自動でやってくれます。
> `try { } finally { fclose($f); }` まで書いたのと同じ安全さです。

---

## 7.5 CSV の書き出し

### 解説

`csv.writer(f, ...)` は、値のリストを渡すと **正しくエスケープした CSV の 1 行** にして書き込んでくれるオブジェクトです。

- `writer.writerow(リスト)` — 1 行書く
- `writer.writerows(リストのリスト)` — 複数行まとめて書く
- `quoting=csv.QUOTE_ALL` — すべての値を `"` で囲む（元システムのエクスポートと同じ形）
- `lineterminator="\r\n"` — 行末を CRLF（Windows 形式）にする

```python
>>> writer.writerow(["ThinkPad", "注文番号:1\nJAN:2"])
# 書かれる内容: "ThinkPad","注文番号:1\nJAN:2"\r\n
```

値の中に改行・カンマ・`"` があっても、`csv` モジュールが正しく囲んでエスケープしてくれます。
**CSV を自分で `",".join(...)` して作ってはいけません**。値にカンマが入った瞬間に壊れます。

### なぜ newline="" が必要か

Python は通常、テキストファイルに書くとき `\n` を OS の改行コードに **自動変換** します（Windows なら `\r\n`）。
`csv` モジュールは改行を自分で管理するので、自動変換があると、Windows で `\r\r\n` のような二重の改行になります。
**CSV を開くときは必ず `newline=""`** — これは Python 公式ドキュメントにも明記されている決まりです。

dummy_csv のコメントにある「値の中の改行は LF のまま」も、`newline=""` で自動変換を止めているからこそ実現できています。

---

## 7.6 CSV の読み込み（題材には無いが頻出）

dummy_csv は書くだけですが、実務では読む方がずっと多いので紹介します。

```python
import csv
from pathlib import Path

path = Path("dummy/sample.csv")
with path.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):     # 1 行目をヘッダとして、各行を辞書で返す
        print(row["品名"], int(row["数量"]))
```

- `csv.DictReader` は各行を `{"品名": "A", "数量": "2"}` のような辞書で返します。**列の位置ではなく列名で読める** ので、
  列の順番が変わっても壊れません（PivotAggregationSystem の設計原則「特定の列名をコードに埋め込まない」にも通じます）。
- 値は **すべて文字列** で返ります。数値として使うなら `int()` / `float()` で変換します。
- Excel が出力した UTF-8 の CSV は先頭に BOM が付くことがあり、そのままだと 1 列目の名前が `"﻿品名"` になります。
  その場合は `encoding="utf-8-sig"` で開くと BOM を取り除いてくれます。

---

## 7.7 ファイルの一覧と削除

### 題材コード

`generate_dummy_csv.py:521-524`

```python
series = re.compile(rf"^{re.escape(args.name)}_?\d{{14}}_\d+\.csv$")
for old in args.out_dir.iterdir():
    if series.match(old.name):
        old.unlink()
```

### 解説

前回出力した分割ファイルを消してから書き出す処理です。

- `iterdir()` — フォルダ直下のファイル・フォルダを 1 つずつ返す
- `old.name` — ファイル名部分だけ（フォルダのパスを含まない）
- `series.match(...)` — 正規表現（第 11 章）で「このスクリプトが作った形の名前か」を判定
- `old.unlink()` — 削除

**消す対象を「自分が作った形の名前」に限定している** のが重要です。`*.csv` を全部消すような書き方だと、
同じフォルダに置いた別の CSV まで消してしまいます。削除処理は、対象を最小限に絞るのが実務の鉄則です。

### glob のパターン

| パターン | 意味 |
|---|---|
| `d.glob("*.md")` | 直下の `.md` |
| `d.glob("**/*.md")` | サブフォルダも含めてすべての `.md` |
| `d.rglob("*.md")` | 同上（再帰の省略形） |

`glob` の返す順番は保証されないので、build_kb は `sorted()` で並べています（第 5 章）。

---

## 7.8 キャッシュの仕組みを読む

ここまでの知識で、`scripts/embedders.py` のキャッシュ処理が読めます。

```python
def _cache_path(self, text: str, input_type: str) -> Path:
    raw = f"{self.model}|{self.dim}|{input_type}|{text}"
    key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{self.model}-{self.dim}" / f"{key}.json"
```

1. 「モデル名 | 次元 | input_type | 本文」をつないだ文字列を作る
2. その SHA-256 ハッシュ（64 桁の 16 進数）をファイル名にする（ハッシュは第 14 章）
3. `.cache/embeddings/<モデル>-<次元>/<ハッシュ>.json` というパスを返す

`embed()` 側では、

- `p.exists()` なら `json.loads(p.read_text())` でキャッシュから読む（API 呼び出しなし = 無料）
- 無ければ API を呼び、`p.parent.mkdir(parents=True, exist_ok=True)` → `p.write_text(json.dumps(vec))` で保存

本文が 1 文字でも変われば、ハッシュが変わって別のファイルになります。
「**同じ入力には同じ結果を、2 回目からはファイルから**」という単純なキャッシュですが、
API の課金を劇的に減らす実務的な仕組みです。

---

## 章末問題

### 問 1（読解）

`Path("/repo/scripts/build_kb.py")` を `p` としたとき、次の値は？

`p.name` / `p.stem` / `p.suffix` / `p.parent.name` / `p.parent.parent / "corpus" / "pages"`

### 問 2（読解）

`json.dumps({"tau": 0.85, "ok": True, "id": None}, ensure_ascii=False)` の結果の文字列は？

### 問 3（書く）

`corpus/pages/` の `.md` ファイルを **ファイル名順** にすべて読み、
`ファイル名: 文字数` を 1 行ずつ表示するスクリプトを書いてください。ファイルのパスはスクリプトの場所を基準にしてください
（スクリプトは `scripts/` に置く想定）。

### 問 4（書く）

次のリストを `out/score.csv` に、すべての値をダブルクォートで囲んだ UTF-8 の CSV として書き出してください。
`out/` フォルダが無ければ作ること。

```python
rows = [["tau", "coverage"], [0.85, "18%"], [0.60, "64%"]]
```

### 問 5（書く）

問 4 で書いた CSV を `csv.DictReader` で読み、`coverage` 列だけをリストで表示してください。

### 問 6（考える）

次のコードの問題点を 2 つ挙げてください。

```python
f = open("data.csv", "w")
f.write(",".join(["品名", "備考"]) + "\n")
f.write(",".join(["ThinkPad", "メモリ 8GB, SSD 256GB"]) + "\n")
```

---

## 回答

<details>
<summary>問 1 の回答</summary>

| 式 | 結果 |
|---|---|
| `p.name` | `'build_kb.py'` |
| `p.stem` | `'build_kb'` |
| `p.suffix` | `'.py'` |
| `p.parent.name` | `'scripts'` |
| `p.parent.parent / "corpus" / "pages"` | `PosixPath('/repo/corpus/pages')` |

</details>

<details>
<summary>問 2 の回答</summary>

`'{"tau": 0.85, "ok": true, "id": null}'`

`True` → `true`、`None` → `null` に変換されます。

</details>

<details>
<summary>問 3 の回答</summary>

```python
"""corpus/pages/*.md の文字数を表示する。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / "corpus/pages"


def main() -> None:
    for path in sorted(PAGES_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        print(f"{path.name}: {len(text)}")


if __name__ == "__main__":
    main()
```

</details>

<details>
<summary>問 4 の回答</summary>

```python
import csv
from pathlib import Path

rows = [["tau", "coverage"], [0.85, "18%"], [0.60, "64%"]]
out = Path("out/score.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerows(rows)
```

数値の `0.85` も `"0.85"` として書かれます（CSV に型はありません）。

</details>

<details>
<summary>問 5 の回答</summary>

```python
import csv
from pathlib import Path

with Path("out/score.csv").open(encoding="utf-8", newline="") as f:
    coverages = [row["coverage"] for row in csv.DictReader(f)]
print(coverages)  # ['18%', '64%']
```

第 5 章の内包表記と組み合わせると 1 行で集められます。

</details>

<details>
<summary>問 6 の回答</summary>

1. **値の中のカンマがエスケープされない**: `"メモリ 8GB, SSD 256GB"` がカンマで 2 列に割れ、列数がずれます。
   `csv.writer` を使えば自動で `"..."` で囲まれます。
2. **ファイルを閉じていない・文字コードと newline を指定していない**: `close()` が無いので、書き込みがディスクに反映されない
   ことがあります。`with open("data.csv", "w", encoding="utf-8", newline="") as f:` と書くべきです。

</details>
