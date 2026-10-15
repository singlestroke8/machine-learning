.PHONY: setup db-up db-down db-reset psql py check-secrets

# 初回: pre-commit 登録、イメージ取得、Python ツールイメージのビルド
setup:
	git config core.hooksPath .githooks
	docker pull pgvector/pgvector:pg16
	docker compose --profile tools build py
	@test -f .env || (cp .env.example .env && echo ".env を作成しました。鍵を記入してください。")

db-up:
	docker compose up -d db
	docker compose exec db sh -c 'until pg_isready -U rag -d rag >/dev/null 2>&1; do sleep 1; done'

db-down:
	docker compose down

# ボリュームを消して db/init/*.sql から再作成
db-reset:
	docker compose down -v
	$(MAKE) db-up

psql:
	docker compose exec db psql -U rag -d rag

# 例: make py ARGS="scripts/build_kb.py --out kb.json"
py:
	docker compose --profile tools run --rm py $(ARGS)

check-secrets:
	scripts/check-secrets.sh --all
