# 第 11 章 正規表現と文字列処理

頻度: ★★★☆☆ — ログ解析・設定ファイルの読み取り・データの整形でよく使います。
PHP の `preg_match` を使ったことがあれば、正規表現そのものの書き方はほぼ同じです。違うのは「呼び出し方」です。

この章で読めるようになるコード:

```python
meta = dict(re.findall(r"<!--\s*([a-z_]+):\s*(.*?)\s*-->", text))
body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
m = re.match(r"Q:\s*(.+?)\s*=>\s*([a-z0-9\-]+)\s*$", line.strip())
_STRIP = re.compile(r"[\s。、．，・！!？?「」『』()（）\[\]〜~\-–—/／…•*※:：;；\"'’”]+")
series = re.compile(rf"^{re.escape(args.name)}_?\d{{14}}_\d+\.csv$")
```

---

## 11.1 raw 文字列 `r"..."`

正規表現はほぼ必ず **`r"..."`（raw 文字列）** で書きます。

```python
>>> print("a\nb")      # \n が改行になる
a
b
>>> print(r"a\nb")     # r を付けると \ をそのまま文字として扱う
a\nb
```

正規表現は `\s`（空白）や `\d`（数字）のように `\` を多用します。通常の文字列だと `\` が Python 側で先に解釈されてしまうので、
`r` を付けて **`\` をそのまま re モジュールに渡す** のが決まりです。

> **PHP との比較**
> PHP は `'/\s+/'` のようにシングルクォートとデリミタ `/.../` で書きます。
> Python は **デリミタが無く**、フラグ（`i` や `s`）は `flags=re.I` のように引数で渡します。

---

## 11.2 re モジュールの関数

| 関数 | 何をするか | 戻り値 | PHP | 題材 |
|---|---|---|---|---|
| `re.match(p, s)` | **先頭から** マッチするか | Match か None | `preg_match('/^.../')` | eval の評価セット読み込み |
| `re.search(p, s)` | **どこかに** マッチするか | Match か None | `preg_match` | eval の設定読み込み |
| `re.fullmatch(p, s)` | **全体が** マッチするか | Match か None | `preg_match('/^...$/')` | |
| `re.findall(p, s)` | すべてのマッチ | リスト | `preg_match_all` | build_kb のメタ情報 |
| `re.sub(p, 置換, s)` | 置換 | 新しい文字列 | `preg_replace` | build_kb, textmatch |
| `re.split(p, s)` | 分割 | リスト | `preg_split` | |
| `re.compile(p)` | パターンを事前に準備 | Pattern | — | textmatch, dummy_csv |

**`match` と `search` の違い** はよく問われます。`match` は「文字列の先頭から」しか探しません。

```python
>>> re.match(r"\d+", "No.123")    # 先頭が数字ではない
None
>>> re.search(r"\d+", "No.123")   # どこかにあれば OK
<re.Match object; span=(3, 6), match='123'>
```

---

## 11.3 match とグループ — 評価セットを読む

### 題材コード

`reference/okubo-shamisen/eval.py:34-45`

```python
def load_eval_sets() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    positives: list[tuple[str, str]] = []
    for line in (EVAL_DIR / "positive.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"Q:\s*(.+?)\s*=>\s*([a-z0-9\-]+)\s*$", line.strip())
        if m:
            positives.append((m.group(1), m.group(2)))
    negatives: list[tuple[str, str]] = []
    for line in (EVAL_DIR / "negative.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"Q:\s*(.+?)\s*\[([a-z\-]+)\]\s*$", line.strip())
        if m:
            negatives.append((m.group(1), m.group(2)))
    return positives, negatives
```

読み込むファイル（`corpus/eval/positive.md`）の中身:

```
# 評価用: 正答すべき質問（言い換えテスト）

想定問答コーパスには含まれない言い回しで、対応するエントリIDに到達できるかを測る。
形式: `Q: 質問文 => 期待するエントリID`

Q: 六十歳から始めるのは遅すぎますか？ => age
Q: 定年後の趣味として考えているのですが => age
```

### パターンを分解する

`Q:\s*(.+?)\s*=>\s*([a-z0-9\-]+)\s*$`

| 部分 | 意味 |
|---|---|
| `Q:` | 文字どおり `Q:` |
| `\s*` | 空白が 0 個以上 |
| `(.+?)` | **グループ 1**: 1 文字以上（`?` で **最短一致**） = 質問文 |
| `\s*=>\s*` | 前後に空白があってもよい `=>` |
| `([a-z0-9\-]+)` | **グループ 2**: 英小文字・数字・ハイフンが 1 文字以上 = FAQ の id |
| `\s*$` | 末尾に空白があってもよく、そこで行が終わる |

```python
>>> m = re.match(r"Q:\s*(.+?)\s*=>\s*([a-z0-9\-]+)\s*$", "Q: 六十歳から始めるのは遅すぎますか？ => age")
>>> m.group(1), m.group(2)
('六十歳から始めるのは遅すぎますか？', 'age')
>>> m.groups()
('六十歳から始めるのは遅すぎますか？', 'age')
```

- **`re.match` は、マッチしなければ `None` を返します**。だから `if m:` で確認してから `m.group()` を使います。
  確認せずに `m.group(1)` と書くと、マッチしない行で `AttributeError: 'NoneType' object has no attribute 'group'` になります。
  これは **Python の正規表現で一番多いエラー** です。
- 見出し行や説明文の行は `Q:` で始まらないので `None` になり、自然に読み飛ばされます。
  説明文の「形式: \`Q: 質問文 => 期待するエントリID\`」も、`re.match` は **先頭からしか見ない** ので対象外になります（`re.search` だったら誤って拾っていたところです）。

### 最短一致 `.+?` と最長一致 `.+`

`(.+?)` の `?` が無いと、`.+` は **できるだけ長く** マッチしようとします。この行では結果は同じですが、
「質問文の中に `=>` が含まれる」ような行では、どこで区切るかが変わります。
**「次の区切りまで」を取りたいときは最短一致 `.+?` / `.*?`** と覚えておくと安全です。

### Python 3.8 以降の書き方: 代入式（セイウチ演算子）

```python
for line in lines:
    if m := re.match(r"Q:\s*(.+?)\s*=>\s*([a-z0-9\-]+)\s*$", line.strip()):
        positives.append((m.group(1), m.group(2)))
```

`:=` は「代入して、その値を条件にも使う」演算子です（形がセイウチの顔に似ているのでこう呼ばれます）。
`m = ...` と `if m:` を 1 行にまとめられます。PHP の `if ($m = ...)` に近い書き方です。

---

## 11.4 findall — 全部取り出す

### 題材コード

`scripts/build_kb.py:25-26`

```python
# 例: <!-- page_key: column-equipment --> → {"page_key": "column-equipment"}
meta = dict(re.findall(r"<!--\s*([a-z_]+):\s*(.*?)\s*-->", text))
```

読み込むファイル（`corpus/pages/*.md`）の先頭:

```
<!-- source: page-column-culture.php -->
<!-- page_key: column-culture -->
<!-- title: コラム：文化と歴史 -->
```

### 解説

`re.findall` の戻り値は、パターン中の **グループの数で形が変わります**。

| グループの数 | 戻り値 |
|---|---|
| 0 個 | マッチした文字列のリスト `['123', '456']` |
| 1 個 | グループ 1 の文字列のリスト |
| 2 個以上 | **グループのタプルのリスト** `[('page_key', 'column-culture'), ('title', '...')]` |

このパターンはグループが 2 つ（キーと値）なので、`[(キー, 値), ...]` が返ります。
第 2 章で見たとおり、それを `dict()` に渡すと辞書になります。**正規表現 + dict() で簡易パーサーを 1 行で書く** テクニックです。

`(.*?)\s*-->` は「`-->` の手前の空白は値に含めない」ための形です。`(.*)` だと最長一致で値の末尾に空白が残ります。

---

## 11.5 sub とフラグ — 置換する

### 題材コード

`scripts/build_kb.py:27-28`

```python
# re.S で改行をまたぐコメントも消す
body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
```

### 解説

`re.sub(パターン, 置換後, 対象)` は、マッチした部分をすべて置き換えた **新しい文字列** を返します（元の文字列は変わらない）。
ここでは HTML コメントを空文字に置き換えて消しています。

**`.` は既定では改行にマッチしません。** `flags=re.S`（`re.DOTALL`）を付けると、改行も含めて「任意の 1 文字」になります。

```python
>>> re.sub(r"<!--.*?-->", "", "a<!--x\ny-->b")              # 改行をまたぐコメントが消えない
'a<!--x\ny-->b'
>>> re.sub(r"<!--.*?-->", "", "a<!--x\ny-->b", flags=re.S)  # 消える
'ab'
```

もう 1 つ、**最短一致 `.*?` であることが重要** です。

```python
>>> re.sub(r"<!--.*-->", "", "a<!--1-->b<!--2-->c", flags=re.S)    # 最長一致
'ac'          # 最初の <!-- から最後の --> までまとめて消え、本文 b も消える！
>>> re.sub(r"<!--.*?-->", "", "a<!--1-->b<!--2-->c", flags=re.S)   # 最短一致
'abc'
```

| フラグ | 意味 | PHP の修飾子 |
|---|---|---|
| `re.I`（`re.IGNORECASE`） | 大文字小文字を区別しない | `i` |
| `re.S`（`re.DOTALL`） | `.` が改行にもマッチ | `s` |
| `re.M`（`re.MULTILINE`） | `^` `$` が各行の先頭・末尾にマッチ | `m` |
| 複数指定 | `flags=re.I \| re.S` | `/.../is` |

### 前後の不要な文字を取る

`reference/okubo-shamisen/eval.py:119`

```python
rid = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", reply, flags=re.I)
```

LLM が `「fee-monthly」。` のように余計な記号を付けて返してきたときに、**前後の英数字以外を取り除く** 処理です。

- `^[^a-z0-9]+` — 先頭の、英数字 **以外**（`[^...]` は「...以外」）の連続
- `|` — または
- `[^a-z0-9]+$` — 末尾の、英数字以外の連続

```python
>>> re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", "「fee-monthly」。", flags=re.I)
'fee-monthly'
```

その後 `any(c["id"] == rid for c in cands)` で、**候補に実在する id かどうか** を確かめています（第 5 章）。
LLM の出力を「整形 → 検証」してから使う、という実務の型です。

---

## 11.6 compile — 何度も使うパターンは事前に準備

### 題材コード

`reference/okubo-shamisen/textmatch.py:11-18`

```python
_STRIP = re.compile(r"[\s。、．，・！!？?「」『』()（）\[\]〜~\-–—/／…•*※:：;；\"'’”]+")


def normalize(text: str) -> str:
    """NFKC正規化 → 小文字化 → 空白・記号除去。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return _STRIP.sub("", text)
```

### 解説

`re.compile(パターン)` は、パターンを解析した **Pattern オブジェクト** を作ります。
`_STRIP.sub(...)`、`_STRIP.match(...)` のように、re の関数と同じ名前のメソッドが使えます。

- `normalize()` は FAQ の照合で何千回も呼ばれるので、パターンを **モジュールの読み込み時に 1 回だけ** 準備しています。
  （実は `re.sub()` も内部でキャッシュするので速度差は小さいのですが、**パターンに名前を付けて意図を示せる** のが大きな利点です。）
- 名前の先頭の `_` は「このモジュールの内部用」（第 6 章）。

パターンの中身は **文字クラス `[...]`** で、「空白・句読点・括弧・記号のどれか」の 1 文字以上の連続です。
`[...]` の中では `(` や `?` は特別な意味を失いますが、`[ ] \ -` などはエスケープが必要なので `\[\]`、`\-` と書かれています。

### Unicode 正規化（NFKC）

`unicodedata.normalize("NFKC", text)` は、**見た目が同じ・意味が同じ文字を 1 つの形に揃える** 処理です。
日本語のテキストを扱うなら必須の知識です。

```python
>>> unicodedata.normalize("NFKC", "ＡＢＣ１２３ｱｲｳ！？（）")
'ABC123アイウ!?()'
```

- 全角英数字 → 半角英数字
- 半角カタカナ → 全角カタカナ
- 全角の記号 → 半角の記号（`！` → `!`）

そのうえで `.lower()` で小文字にし、`_STRIP` で記号と空白を消します。

```python
>>> normalize("体験レッスンは、ありますか？ (初心者)")
'体験レッスンはありますか初心者'
>>> normalize("ＯＫ！")
'ok'
```

これで「全角・半角」「句読点の有無」「空白の有無」が違うだけの質問が、同じ文字列として比べられるようになります。
docstring にある「ウィジェット（JS）とサーバ（PHP）に移植する際はこのファイルと挙動を一致させる」は、
JS では `str.normalize("NFKC")`、PHP では `Normalizer::normalize($s, Normalizer::FORM_KC)` に当たります。

---

## 11.7 f 文字列で正規表現を組み立てる

### 題材コード

`PivotAggregationSystem/scripts/generate_dummy_csv.py:521`

```python
series = re.compile(rf"^{re.escape(args.name)}_?\d{{14}}_\d+\.csv$")
```

`reference/okubo-shamisen/eval.py:57-58`

```python
for key in ("tau0", "tau1", "tau1_cand"):
    m = re.search(rf"'{key}'\s*=>\s*([0-9.]+)", text)
```

### 解説

`rf"..."` は **raw 文字列 + f 文字列** です。正規表現の中に変数を埋め込めます。

注意点が 2 つあります。

1. **変数の中身は `re.escape()` で必ずエスケープする**。
   `args.name` には `【業務】販売管理DB：管理者用` のような文字列が入ります。もし `.` や `(` を含むファイル名だったら、
   正規表現の記号として解釈されてしまいます。`re.escape()` は記号の前に `\` を付けて「ただの文字」にしてくれます
   （PHP の `preg_quote` に当たる）。
2. **正規表現の `{14}`（ちょうど 14 回）は `{{14}}` と書く**。
   f 文字列では `{ }` が変数の埋め込みに使われるので、文字としての `{` `}` は 2 つ重ねて書きます。

組み立てた結果のパターンは次のとおりです。

```
^【業務】販売管理DB：管理者用_?\d{14}_\d+\.csv$
```

「名前 + 任意の `_` + 14 桁の数字（出力日時）+ `_` + 連番 + `.csv`」— dummy_csv が自分で出力した分割ファイルだけに一致し、
それ以外のファイルを誤って消さないための判定です（第 7 章）。

eval.py の方は、PHP の設定ファイル `config.php` の `'tau0' => 0.85,` という行から値を取り出しています。
**別言語の設定ファイルを正規表現で読み、評価を本番の設定に自動で合わせる** という実務的な工夫です。

```python
>>> re.search(r"'tau0'\s*=>\s*([0-9.]+)", "'tau0' => 0.85,").group(1)
'0.85'
```

取り出した値は文字列なので、`float(m.group(1))` で数値にしています（eval.py:61）。

---

## 11.8 正規表現の早見表

| 書き方 | 意味 |
|---|---|
| `.` | 任意の 1 文字（改行以外。`re.S` で改行も） |
| `\d` / `\D` | 数字 / 数字以外（Python 3 の `\d` は全角数字にもマッチする） |
| `\s` / `\S` | 空白（スペース・タブ・改行・全角スペース） / 空白以外 |
| `\w` / `\W` | 単語の文字（英数字・`_`・**日本語の文字も含む**） / それ以外 |
| `[abc]` / `[^abc]` | a か b か c / それ以外 |
| `[a-z0-9]` | 範囲 |
| `*` `+` `?` | 0 回以上 / 1 回以上 / 0 か 1 回（最長一致） |
| `*?` `+?` | 最短一致 |
| `{14}` `{1,6}` | ちょうど 14 回 / 1〜6 回 |
| `^` `$` | 先頭 / 末尾 |
| `(...)` | グループ（取り出せる） |
| `(?:...)` | グループ（取り出さない、まとめるだけ） |
| `(?P<name>...)` | 名前付きグループ（`m.group("name")` で取り出す） |
| `a\|b` | a または b |

build_kb.py の見出し判定 `re.match(r"^#{1,6} ", line)` は「行頭に `#` が 1〜6 個、続いて半角スペース」。
Markdown の見出し（`# ` 〜 `###### `）だけに一致し、`#タグ` のようなものには一致しません。

### 正規表現を使わない方がよい場面

- 単純な部分一致 → `"在庫生産" in name`
- 接頭辞・接尾辞 → `name.startswith("#")` / `path.endswith(".csv")`
- 単純な置換 → `s.replace("{{SCHOOL}}", school_name)`（eval.py:86）

**文字列メソッドで書けるならそちらが速くて読みやすい** です。正規表現は「パターン」が必要なときだけ使いましょう。

---

## 章末問題

### 問 1（読解）

次の結果は？

```python
re.match(r"\d+", "2026年10月")
re.match(r"\d+", "令和8年")
re.search(r"\d+", "令和8年")
re.findall(r"\d+", "C-000000047 と 2026年")
re.findall(r"(\w+)=(\d+)", "tau0=85 tau1=90")
```

### 問 2（書く）

`corpus/eval/negative.md` の行 `Q: 今日の東京の天気を教えて [out-of-scope]` から、
質問文と分類（`out-of-scope`）を取り出すコードを、**代入式 `:=`** を使って書いてください。

### 問 3（書く）

設定ファイルの文字列 `text = "'router_k' => 3,\n'router_model' => 'claude-haiku-4-5',"` から、
`router_k` を **整数** で、`router_model` を文字列で取り出してください。見つからないときは `router_k` を 3、`router_model` を `""` にしてください。

### 問 4（読解）

`textmatch.normalize()` に次の文字列を渡した結果は？

1. `"体験　レッスン、ありますか？"`（間に全角スペース）
2. `"ＡＩ　チャット"`
3. `"三味線（津軽）"`

### 問 5（バグ探し）

次のコードは、ユーザーが入力した検索語でファイル名を絞り込むものです。何が問題ですか？

```python
keyword = input("検索語: ")      # 例: "price(old)"
pattern = re.compile(f"^{keyword}.*\\.md$")
```

### 問 6（書く）

Markdown の本文から、`[表示テキスト](URL)` 形式のリンクをすべて `(表示テキスト, URL)` のリストで取り出してください。

```python
md = "詳しくは[料金ページ](/price)と[アクセス](/access)をご覧ください。"
# → [('料金ページ', '/price'), ('アクセス', '/access')]
```

---

## 回答

<details>
<summary>問 1 の回答</summary>

| 式 | 結果 |
|---|---|
| `re.match(r"\d+", "2026年10月")` | Match（`'2026'`） |
| `re.match(r"\d+", "令和8年")` | `None`（先頭が数字ではない） |
| `re.search(r"\d+", "令和8年")` | Match（`'8'`） |
| `re.findall(r"\d+", "C-000000047 と 2026年")` | `['000000047', '2026']` |
| `re.findall(r"(\w+)=(\d+)", "tau0=85 tau1=90")` | `[('tau0', '85'), ('tau1', '90')]` |

</details>

<details>
<summary>問 2 の回答</summary>

```python
import re

line = "Q: 今日の東京の天気を教えて [out-of-scope]"
if m := re.match(r"Q:\s*(.+?)\s*\[([a-z\-]+)\]\s*$", line.strip()):
    question, category = m.groups()
    print(question, category)   # 今日の東京の天気を教えて out-of-scope
```

`m.groups()` はすべてのグループをタプルで返すので、アンパックで受け取れます。

</details>

<details>
<summary>問 3 の回答</summary>

```python
import re

text = "'router_k' => 3,\n'router_model' => 'claude-haiku-4-5',"

m = re.search(r"'router_k'\s*=>\s*([0-9]+)", text)
router_k = int(m.group(1)) if m else 3

m = re.search(r"'router_model'\s*=>\s*'([^']+)'", text)
router_model = m.group(1) if m else ""

print(router_k, router_model)   # 3 claude-haiku-4-5
```

eval.py:62-65 とほぼ同じ形です。`[^']+` は「`'` 以外の 1 文字以上」で、閉じクォートの手前までを取ります。

</details>

<details>
<summary>問 4 の回答</summary>

1. `'体験レッスンありますか'` — 全角スペースは NFKC で半角スペースになり、`\s` で消える。読点・疑問符も消える
2. `'aiチャット'` — 全角英字は半角になり、`.lower()` で小文字に
3. `'三味線津軽'` — 全角括弧は半角 `()` になり、`_STRIP` で消える

</details>

<details>
<summary>問 5 の回答</summary>

`keyword` をエスケープせずに正規表現に埋め込んでいます。`price(old)` の `(` `)` はグループとして解釈され、
「`price(old)` という文字列」ではなく「`priceold`」にマッチするパターンになってしまいます。
`[` だけを入力されると `re.error` で落ちます。

```python
pattern = re.compile(rf"^{re.escape(keyword)}.*\.md$")
```

ユーザー入力や外部の値を正規表現に埋め込むときは、**必ず `re.escape()`**。SQL のパラメータ化と同じ考え方です。

</details>

<details>
<summary>問 6 の回答</summary>

```python
import re

md = "詳しくは[料金ページ](/price)と[アクセス](/access)をご覧ください。"
links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", md)
print(links)   # [('料金ページ', '/price'), ('アクセス', '/access')]
```

- `\[` `\]` `\(` `\)` — 括弧そのもの（エスケープ）
- `([^\]]+)` — `]` 以外の 1 文字以上（表示テキスト）
- `([^)]+)` — `)` 以外の 1 文字以上（URL）

`(.+?)` の最短一致でも書けますが、「閉じ括弧以外」と書く方が意図が明確で、想定外のマッチも起きにくくなります。

</details>
