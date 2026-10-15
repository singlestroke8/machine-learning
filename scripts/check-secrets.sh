#!/usr/bin/env sh
# コミット対象に API キーらしき文字列が混ざっていないか確認する。
# 使い方: scripts/check-secrets.sh          （ステージ済みの差分を検査）
#         scripts/check-secrets.sh --all    （追跡ファイル全体を検査）
set -eu

pattern='sk-ant-[A-Za-z0-9_-]{20,}|pa-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|aws_secret_access_key|-----BEGIN [A-Z ]*PRIVATE KEY-----'

# このスクリプト自身（パターン定義を含む）は検査対象から外す
self='scripts/check-secrets.sh'
if [ "${1:-}" = "--all" ]; then
  hits=$(git ls-files -z -- . ":(exclude)$self" | xargs -0 grep -nE "$pattern" -- 2>/dev/null || true)
else
  hits=$(git diff --cached -U0 -- . ":(exclude)$self" | grep -E '^\+' | grep -vE '^\+\+\+' | grep -nE "$pattern" || true)
fi

if [ -n "$hits" ]; then
  echo "鍵らしき文字列が見つかりました。コミットを中止します:" >&2
  echo "$hits" >&2
  exit 1
fi

if git diff --cached --name-only | grep -qxE '\.env(\..*)?' ; then
  if ! git diff --cached --name-only | grep -qx '.env.example'; then
    echo ".env がステージされています。コミットを中止します。" >&2
    exit 1
  fi
fi
exit 0
