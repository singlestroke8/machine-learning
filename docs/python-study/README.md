# 実コードで学ぶ Python — 実務 2 年レベルへの道

このリポジトリと PivotAggregationSystem にある **実際に動いている Python スクリプト** を教材に、
Python の文法と実務の書き方を学ぶ教材です。

- 想定読者: JavaScript はそこそこ書ける / PHP（Laravel・CakePHP）・HTML・CSS は得意 / Python は基本だけ
- ゴール: 実務 2 年目くらいのコードを **読めて、直せて、同じ水準で書ける**
- 範囲外: FastAPI・Django などの Web フレームワーク（土台の文法を固めてから）

## 題材にするコード

| 略称 | ファイル | 何をするか | 行数 |
|---|---|---|---|
| build_kb | `scripts/build_kb.py` | Markdown のページを見出しごとの「チャンク」に分けて JSON に書く | 110 |
| embedders | `scripts/embedders.py` | テキストを Voyage API でベクトル化し、結果をファイルにキャッシュ | 108 |
| load_kb | `scripts/load_kb.py` | チャンクとベクトルを Postgres に投入 | 67 |
| export_kb | `scripts/export_kb.py` | DB から読み出して `kb.json` に書き出す | 80 |
| textmatch | `reference/okubo-shamisen/textmatch.py` | 文字バイグラムで文章の似ている度合いを測る | 41 |
| ref_embedders | `reference/okubo-shamisen/embedders.py` | 継承を使った埋め込みクラス群（本番ボット版） | 135 |
| ref_build_kb | `reference/okubo-shamisen/build_kb.py` | 本番ボット版の KB ビルド | 171 |
| eval | `reference/okubo-shamisen/eval.py` | 検索精度の評価と閾値の掃引 | 417 |
| dummy_csv | `PivotAggregationSystem/scripts/generate_dummy_csv.py` | 販売管理のダミー CSV を生成（標準ライブラリのみ） | 544 |

コード中のパスは、特に断りがなければ `machine-learning/` からの相対パスです。
dummy_csv だけは `PivotAggregationSystem/` リポジトリにあります。

## 章立て（実務での使用頻度順）

★ の数は「実務のコードでどれだけ頻繁に出会うか」の目安です。

| 章 | テーマ | 頻度 | 主な題材 |
|---|---|---|---|
| [01](01-variables-and-strings.md) | 変数・データ型・文字列（f 文字列） | ★★★★★ | build_kb, eval |
| [02](02-collections.md) | リスト・辞書・タプル・セット | ★★★★★ | embedders, dummy_csv |
| [03](03-control-flow.md) | 制御構文（if / for / enumerate / zip / 条件式） | ★★★★★ | build_kb, embedders |
| [04](04-functions.md) | 関数（引数・戻り値・アンパック・クロージャ・lambda） | ★★★★★ | build_kb, eval |
| [05](05-comprehensions-and-builtins.md) | 内包表記・ジェネレータ式・組み込み関数 | ★★★★☆ | eval, export_kb |
| [06](06-modules-and-imports.md) | モジュールと import・`__name__` | ★★★★☆ | load_kb, export_kb |
| [07](07-files-json-csv.md) | ファイル入出力（pathlib / json / csv） | ★★★★☆ | build_kb, dummy_csv |
| [08](08-exceptions.md) | 例外処理 | ★★★★☆ | embedders, ref_embedders |
| [09](09-type-hints.md) | 型ヒント | ★★★★☆ | 全ファイル |
| [10](10-classes.md) | クラスとオブジェクト指向 | ★★★☆☆ | embedders, ref_embedders, eval |
| [11](11-regex-and-text.md) | 正規表現と文字列処理 | ★★★☆☆ | build_kb, textmatch, eval |
| [12](12-with-and-database.md) | with 文・コンテキストマネージャ・DB 接続 | ★★★☆☆ | load_kb, export_kb |
| [13](13-argparse-cli.md) | argparse でコマンドラインツールを作る | ★★★☆☆ | dummy_csv, eval |
| [14](14-stdlib.md) | 標準ライブラリ（datetime / decimal / random / hashlib / itertools） | ★★★☆☆ | dummy_csv, embedders |
| [15](15-towards-practice.md) | 実務への橋渡し（dataclass・logging・pytest・環境構築）と総合演習 | ★★★★☆ | — |

## 各章の構成

1. **題材コード** — 実際のファイルから抜粋（`ファイル:行番号` 付き）
2. **解説** — 1 行ずつ、なぜそう書くのかまで
3. **他言語との比較** — JS / PHP と違うところだけ短く
4. **落とし穴** — 実務でよく踏むもの
5. **章末問題 & 回答** — 回答は折りたたみ（`<details>`）。まず自分で考えてから開いてください

## 進め方

1. 章を読む前に、題材のファイルを一度ざっと開いて眺める（分からなくて構いません）
2. 章を読みながら、コード片を対話モード（REPL）で実際に打って確かめる
3. 章末問題を解く。書く問題は実際にファイルに書いて実行する
4. 章を終えたら、もう一度題材のファイルを開く。読める範囲が広がっているはずです

### コードの試し方

教材のコード片は標準ライブラリだけで動きます（Python 3.10 以上）。

```bash
python3            # 対話モード（REPL）が起動する。>>> の後に 1 行ずつ打って Enter
python3 -q         # 起動メッセージを出さない
python3 try.py     # ファイルに書いたものを実行
```

ホスト環境を汚したくない場合は Docker で使い捨ての Python を起動できます。

```bash
docker run --rm -it python:3.13 python3
```

REPL では式を打つだけで値が表示されます（`print` 不要）。`help(str.split)` や `dir([])` で
メソッドの一覧と説明も見られます。迷ったらまず REPL で試す、が Python 上達の近道です。
