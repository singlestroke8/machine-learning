---
name: safe-commit
description: 鍵の混入チェックを通してからコミットし、ブランチを push して PR を更新する手順。コミット・push・PR を求められたときに使う。
---

# safe-commit

1. `git status --short` でステージ対象を確認する。`.env`、`logs/`、`node_modules/`、`dist/`、`*.zip` が含まれていたら外す。
2. `scripts/check-secrets.sh --all` を実行し、終了コード 0 を確認する。見つかった場合は該当行を取り除くまでコミットしない。
3. `git add` は対象ファイルを明示する（`git add -A` は使わない）。
4. コミットメッセージは日本語で、1 行目に何をしたかを書き、本文に変更点を箇条書きにする。
5. 作業ブランチへ `git push -u origin <branch>`。main へは push しない。
6. 開いている PR がなければドラフト PR を作る。あれば PR 本文の変更点を更新する。
