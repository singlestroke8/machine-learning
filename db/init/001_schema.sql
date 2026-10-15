-- pgvector 拡張と、ページ本文チャンクのテーブル。
-- embedding は 256 次元（Voyage の output_dimension=256）。
-- model 列でモデル別に同じチャンクを保持し、Embedding 比較に使う。
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
  id         serial PRIMARY KEY,
  page_key   text        NOT NULL,
  title      text        NOT NULL,
  heading    text,
  text       text        NOT NULL,
  model      text        NOT NULL,
  embedding  vector(256) NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_model_idx ON chunks (model);
