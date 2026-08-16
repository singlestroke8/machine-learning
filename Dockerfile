# =============================================================================
# 推論API用のイメージ。
#
# 学習用の依存（MLflow / Optuna / matplotlib）は入れない。推論に不要な
# ライブラリを本番イメージに含めるのは、サイズだけでなく脆弱性の対象面を
# 増やすという意味でも損になる。学習は開発環境か、別のジョブで行う。
#
# ビルドと実行でステージを分け、実行イメージにはビルドツールを残さない。
# =============================================================================

# --- ビルドステージ -----------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# uv 公式イメージからバイナリだけを取り出す（インストールスクリプトを
# 走らせるより、バージョンが固定できて再現性が高い）
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 依存だけを先に解決してレイヤーに固定する。
# アプリのコードを変えただけでライブラリを再インストールしないための定石。
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --no-editable

COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# --- 実行ステージ -------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# LightGBM は OpenMP のランタイムを必要とする
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# root で動かさない
RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DFC_MODEL_PATH=/app/models/model.joblib

USER app
EXPOSE 8000

# モデルが読めない状態を「起動失敗」ではなく「degraded」として扱うため、
# ヘルスチェックは model_loaded まで見る
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, json, sys; \
body = json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); \
sys.exit(0 if body.get('model_loaded') else 1)"

CMD ["uvicorn", "demand_forecast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
