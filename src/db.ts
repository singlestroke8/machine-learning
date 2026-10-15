import pg from "pg";

// pg の Pool: 接続を使い回す入れ物。質問のたびに TCP 接続を張り直さずに済む。
// 接続先は .env の DATABASE_URL（docker-compose の Postgres に localhost:5432 で入る）。
// 呼び出し側（search.ts など）が process.loadEnvFile() を済ませてから最初の query が走る前提で、
// Pool は遅延生成にしている（import した瞬間に環境変数を読むと、読み込み前の空値を掴むため）。
let pool: pg.Pool | undefined;

function getPool(): pg.Pool {
  pool ??= new pg.Pool({ connectionString: process.env.DATABASE_URL, max: 4 });
  return pool;
}

// 検索結果 1 件の型。DB の列名（snake_case）をそのまま使い、変換の手間とズレを避ける
export type ChunkHit = {
  id: number;
  page_key: string;
  title: string;
  heading: string | null; // ページ冒頭のチャンクは見出しなし（NULL）
  text: string;
  score: number; // コサイン類似度（1 に近いほど似ている）
};

export type SearchOptions = {
  model: string; // どのモデルで作ったベクトルと比べるか。質問の埋め込みと必ず同じにする
  topK?: number;
};

// 質問ベクトルに近いチャンクを topK 件返す。
// async 関数なので戻り値は Promise。呼び出し側は await で結果（配列）を受け取る
export async function searchChunks(
  queryVec: number[],
  { model, topK = 12 }: SearchOptions,
): Promise<ChunkHit[]> {
  // pgvector はベクトルを "[0.1,0.2,...]" という文字列で受け取れる。JSON.stringify がちょうどその形になる
  const vecLiteral = JSON.stringify(queryVec);

  // <=> はコサイン距離（0 = 同じ向き、2 = 正反対）。類似度は 1 - 距離で求める。
  // 値は $1, $2... のプレースホルダで渡し、SQL 文字列に直接埋め込まない（SQL インジェクション対策）
  const sql = `
    SELECT id, page_key, title, heading, text,
           1 - (embedding <=> $1::vector) AS score
      FROM chunks
     WHERE model = $2
     ORDER BY embedding <=> $1::vector
     LIMIT $3`;

  // query<ChunkHit> で rows の型を指定する。実際の列と合っているかは TS では検査されないので、
  // SELECT の列名と ChunkHit の項目名を揃えておくのが書き手の責任になる
  const res = await getPool().query<ChunkHit>(sql, [vecLiteral, model, topK]);
  return res.rows;
}

// CLI から使ったときは最後に呼ぶ。Pool が接続を持ったままだと Node のプロセスが終わらないため
export async function closePool(): Promise<void> {
  await pool?.end();
  pool = undefined;
}