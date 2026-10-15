-- HNSW インデックス（コサイン距離）。
-- Phase 3 で全件比較との一致率と速度を比べるために用意する。
-- 初期化時に作成されるが、計測で「索引なし」を見たいときは DROP INDEX して比較する。
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
