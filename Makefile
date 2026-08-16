# 開発でよく打つコマンドをまとめたもの。
# CI と同じコマンドをローカルでも打てるようにして、
# 「手元では通るのに CI で落ちる」を減らす狙い。

.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test test-all check data train figures forecast serve docker-build docker-run clean

help: ## このヘルプを表示する
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## 依存関係をインストールする（学習用も含む）
	uv sync --extra train

lint: ## 静的解析（ruff）
	uv run ruff check .
	uv run ruff format --check .

format: ## 自動整形と自動修正
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## 型検査（mypy strict）
	uv run mypy

test: ## テスト（時間のかかるものを除く）
	uv run pytest -m "not slow"

test-all: ## テスト（全件、カバレッジ付き）
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test ## CI と同じ検査を一括で実行する

data: ## 合成需要データを生成する
	uv run dfc generate-data

train: ## 学習・評価・モデル保存
	uv run dfc train

figures: ## 学習結果から図を生成する
	uv run dfc figures

forecast: ## 保存済みモデルで動作確認の予測を出す
	uv run dfc forecast

serve: ## 推論APIをローカルで起動する
	uv run dfc serve --reload

docker-build: ## 推論APIのイメージをビルドする
	docker build -t demand-forecast-api:local .

docker-run: ## 推論APIをコンテナで起動する
	docker compose up --build

clean: ## 生成物とキャッシュを削除する
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
