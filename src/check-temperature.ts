import Anthropic from "@anthropic-ai/sdk";
import { existsSync } from "node:fs";
import { MODELS } from "./pricing.ts";

if (existsSync(".env")) process.loadEnvFile(".env");
const client = new Anthropic();

// 判定段と同じ条件（temperature 0 / max_tokens 16）で、モデルごとに通るかだけ見る
// ストリーミング不要なので create()。await で応答全体を待つ
for (const model of [MODELS.generate, MODELS.judge]) {
  try {
    const res = await client.messages.create({
      model,
      max_tokens: 16,
      temperature: 0,
      messages: [{ role: "user", content: "「はい」とだけ答えて" }],
    });
    // content はブロックの配列。type で絞り込んでから .text を読む
    const text = res.content.find((b) => b.type === "text")?.text;
    console.log(`${model}: OK  "${text}"  (in=${res.usage.input_tokens} out=${res.usage.output_tokens})`);
  } catch (err) {
    // API エラーは型付き例外。status とメッセージだけ出して次のモデルへ進む
    if (err instanceof Anthropic.APIError) {
      console.log(`${model}: ${err.status} ${err.message}`);
    } else {
      throw err;
    }
  }
}