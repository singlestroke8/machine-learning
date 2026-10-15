import Anthropic from "@anthropic-ai/sdk";
import { existsSync } from "node:fs";
import { MODELS, costJpy } from "./pricing.ts";

// .env を Node 標準機能で読む（dotenv を足さずに済む）。鍵はコードにもチャットにも出さない
if (existsSync(".env")) process.loadEnvFile(".env");

const question = process.argv[2];
if (!question) {
  console.error('usage: npx tsx src/llm.ts "質問"');
  process.exit(1);
}

// ANTHROPIC_API_KEY は環境変数から自動で読まれる
const client = new Anthropic();

// トップレベル await: ESM（"type": "module"）なので main 関数で包まなくてよい
const t0 = performance.now();
let ttftMs: number | null = null;

// stream() はすぐに返るストリームオブジェクト。HTTP の SSE を SDK が型付きイベントに変換してくれる
const stream = client.messages.stream({
  model: MODELS.generate,
  // 費用の上限として効く。思考トークンもここに含まれるので小さくしすぎない
  max_tokens: 4096,
  // Sonnet 5.5 は既定で adaptive thinking。FAQ 程度なら effort を下げて TTFT とコストを抑える
  output_config: { effort: "low" },
  messages: [{ role: "user", content: question }],
});

// for await: SSE のイベントが届くたびにループが 1 回まわる（全文を待たない）
for await (const event of stream) {
  // event.type → event.delta.type の順に絞り込むと、TS が event.delta.text の存在を推論できる
  // 思考ブロック（thinking_delta）は表示しないので、本文の text_delta だけを拾う
  if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
    ttftMs ??= performance.now() - t0; // 最初の 1 文字が来た時刻だけを記録
    process.stdout.write(event.delta.text); // 改行なしで逐次出す
  }
}

// ストリームを読み終えたら、usage を含む完成形のメッセージを受け取る
const message = await stream.finalMessage();
const totalMs = performance.now() - t0;

if (message.stop_reason === "refusal") {
  console.error("\n[refusal]", message.stop_details);
}

const u = message.usage;
const usage = {
  inputTokens: u.input_tokens,
  outputTokens: u.output_tokens, // 思考トークンも出力として課金される
  cacheReadTokens: u.cache_read_input_tokens ?? 0, // キャッシュ未使用時は null になりうる
};

console.log(
  `\n\n---\nmodel=${message.model} stop=${message.stop_reason}` +
    `\nin=${usage.inputTokens} out=${usage.outputTokens} cache_read=${usage.cacheReadTokens}` +
    `\ncost=¥${costJpy(MODELS.generate, usage).toFixed(3)}` +
    ` ttft=${ttftMs?.toFixed(0) ?? "-"}ms total=${totalMs.toFixed(0)}ms`,
);