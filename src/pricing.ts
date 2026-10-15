// モデル ID と料金を 1 か所に集約する。数値は公式料金ページで確認した日付と一緒に管理する。
// 確認日: 2026-10-15

// as const で文字列リテラル型にする（string に広がらないので、存在しない ID はコンパイルエラーになる）
export const MODELS = {
  generate: "claude-sonnet-5-5",
  judge: "claude-haiku-4-5",
} as const;

export type ModelId = (typeof MODELS)[keyof typeof MODELS];

// 単位: USD / 100 万トークン
type Price = { input: number; output: number; cacheRead: number };

// satisfies で「全モデル分そろっているか」を検査しつつ、各値の型は具体的なまま残す
export const PRICES = {
  "claude-sonnet-5-5": { input: 2, output: 10, cacheRead: 0.1 },
  "claude-haiku-4-5": { input: 1, output: 5, cacheRead: 0.1 },
} satisfies Record<ModelId, Price>;

// 為替は計測日の値に更新する（README の表に適用レートを併記する）
export const USD_JPY = 150;

export type Usage = {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
};

// キャッシュ書き込み（入力単価の 1.25 倍）はキャッシュを入れるフェーズで足す
export function costJpy(model: ModelId, u: Usage): number {
  const p = PRICES[model];
  const usd =
    (u.inputTokens * p.input +
      u.outputTokens * p.output +
      u.cacheReadTokens * p.cacheRead) /
    1_000_000;
  return usd * USD_JPY;
}