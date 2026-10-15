import { existsSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { searchChunks, closePool, type ChunkHit } from "./db.ts";

// 埋め込みの条件は scripts/embedders.py と必ず揃える（モデル・次元が違うと比べる意味がない）
export const EMBED_MODEL = "voyage-3.5-lite";
const EMBED_DIM = 256;
const VOYAGE_URL = "https://api.voyageai.com/v1/embeddings";

// Voyage の応答のうち使う部分だけを型にする。API 全体を型付けする必要はない
type VoyageResponse = {
  data: { embedding: number[]; index: number }[];
  usage: { total_tokens: number };
};

// 質問文を 256 次元のベクトルにする。
// 本文チャンクは "document"、質問は "query" で埋め込む（非対称）。同じ文でも向きが変わるため、ここを間違えると精度が落ちる。
// Voyage の公式 TS SDK は使わず fetch で呼ぶ。1 エンドポイントだけなので依存を増やさない（Lambda のサイズも小さく保てる）
export async function embedQuery(text: string): Promise<{ vec: number[]; tokens: number }> {
  const apiKey = process.env.VOYAGE_API_KEY;
  if (!apiKey) throw new Error("VOYAGE_API_KEY が未設定です（.env を確認）");

  // fetch は Node 18+ の標準機能。await で HTTP 応答が届くまで待つ
  const res = await fetch(VOYAGE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({
      input: [text],
      model: EMBED_MODEL,
      input_type: "query",
      output_dimension: EMBED_DIM,
    }),
  });
  // fetch は 4xx/5xx でも例外を投げないので、自分で ok を確認する
  if (!res.ok) throw new Error(`Voyage ${res.status}: ${await res.text()}`);

  // res.json() の戻り値は any なので、上で定義した型として扱う（中身の検査はしない）
  const body = (await res.json()) as VoyageResponse;
  return { vec: body.data[0].embedding, tokens: body.usage.total_tokens };
}

// 質問 → 埋め込み → DB 検索。Phase 4 の API からもこの関数を呼ぶ
export async function search(question: string, topK = 12): Promise<ChunkHit[]> {
  const { vec } = await embedQuery(question);
  return searchChunks(vec, { model: EMBED_MODEL, topK });
}

// ---- ここから CLI: npx tsx src/search.ts "質問" ----
// このファイルを直接実行したときだけ動かす（import されたときは動かない）
const isMain = import.meta.url === pathToFileURL(process.argv[1] ?? "").href;

if (isMain) {
  if (existsSync(".env")) process.loadEnvFile(".env");

  const question = process.argv[2];
  if (!question) {
    console.error('usage: npx tsx src/search.ts "質問" [topK]');
    process.exit(1);
  }
  const topK = Number(process.argv[3] ?? 5);

  // 埋め込みと検索の時間を分けて測る。どちらが遅いかを後で比べるため
  const t0 = performance.now();
  const { vec, tokens } = await embedQuery(question);
  const t1 = performance.now();
  const hits = await searchChunks(vec, { model: EMBED_MODEL, topK });
  const t2 = performance.now();

  hits.forEach((h, i) => {
    // 改行を空白にして先頭 50 文字だけ見せる
    const snippet = h.text.replace(/\s+/g, " ").slice(0, 50);
    console.log(`${i + 1}. ${h.score.toFixed(3)}  [${h.page_key}] ${h.heading ?? "(見出しなし)"}`);
    console.log(`     ${snippet}…`);
  });
  console.log(
    `\nmodel=${EMBED_MODEL} tokens=${tokens}` +
      ` embed=${(t1 - t0).toFixed(0)}ms db=${(t2 - t1).toFixed(0)}ms`,
  );

  // try/finally にせず最後に閉じる。途中で例外なら Node がエラー表示して終了するので十分
  await closePool();
}