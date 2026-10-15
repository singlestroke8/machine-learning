#!/usr/bin/env python3
"""知識ベースの評価と閾値チューニング。

使い方:
    python3 rag/build_kb.py --provider mock   # 先にKBを作る
    python3 rag/eval.py --provider mock       # 同じproviderで評価

測るもの:
- Tier0（バイグラム・ファジーマッチ）: 正答セットの到達率と、拒否セットの誤回答率
- Tier1（埋め込み類似）: Recall@1 / @3 と、閾値ごとのカバー率・誤回答率
- Tier2（チャンク検索）: 正答/拒否セットそれぞれのベストスコア分布（τ2の当たり付け）
- 閾値の推奨値: 拒否セットを1件も誤回答しない範囲で正答カバー率が最大になる点

出力: 標準出力 + rag/eval-report.md
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedders import make_embedder
from textmatch import bigram_cosine

RAG_DIR = Path(__file__).resolve().parent
EVAL_DIR = RAG_DIR / "corpus/eval"


def load_eval_sets() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    positives: list[tuple[str, str]] = []
    for line in (EVAL_DIR / "positive.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"Q:\s*(.+?)\s*=>\s*([a-z0-9\-]+)\s*$", line.strip())
        if m:
            positives.append((m.group(1), m.group(2)))
    negatives: list[tuple[str, str]] = []
    for line in (EVAL_DIR / "negative.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"Q:\s*(.+?)\s*\[([a-z\-]+)\]\s*$", line.strip())
        if m:
            negatives.append((m.group(1), m.group(2)))
    return positives, negatives


CHATBOT_DIR = RAG_DIR.parent / "html/wordpress/wp-content/themes/okubo-shamisen/inc/chatbot"
CONFIG_PHP = CHATBOT_DIR / "config.php"
ROUTER_PROMPT = CHATBOT_DIR / "router-prompt.txt"


def load_deployed_config() -> dict:
    """config.php に書かれた本番の閾値・判定段の設定を読む（評価を本番設定に合わせるため）。"""
    text = CONFIG_PHP.read_text(encoding="utf-8")
    cfg: dict = {}
    for key in ("tau0", "tau1", "tau1_cand"):
        m = re.search(rf"'{key}'\s*=>\s*([0-9.]+)", text)
        if not m:
            sys.exit(f"{CONFIG_PHP} から {key} を読めません")
        cfg[key] = float(m.group(1))
    m = re.search(r"'router_k'\s*=>\s*([0-9]+)", text)
    cfg["router_k"] = int(m.group(1)) if m else 3
    m = re.search(r"'router_model'\s*=>[^\n]*'(claude-[^']+)'", text)
    cfg["router_model"] = m.group(1) if m else ""
    m = re.search(r"'school_name'\s*=>\s*'([^']+)'", text)
    cfg["school_name"] = m.group(1) if m else ""
    return cfg


class Router:
    """engine.php の route() と同じ指示文・同じ入力形式で、候補からFAQを選ばせる。

    結果は rag/.cache/router/ にキャッシュし、同じ質問・候補での再課金を防ぐ。
    """

    def __init__(self, model: str, school_name: str):
        try:
            import anthropic
        except ImportError:
            sys.exit("--router には Anthropic SDK が必要です: pip install anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("--router には ANTHROPIC_API_KEY が必要です")
        self.client = anthropic.Anthropic()
        self.model = model
        self.system = ROUTER_PROMPT.read_text(encoding="utf-8").replace("{{SCHOOL}}", school_name)
        self.calls = 0
        self.cache_dir = RAG_DIR / ".cache/router"

    def __call__(self, question: str, cands: list[dict]) -> str | None:
        items = "".join(
            f'<candidate id="{c["id"]}">\n<examples>{" / ".join(c["questions"][:3])}</examples>\n'
            f'<answer>{c["answer"]}</answer>\n</candidate>\n'
            for c in cands
        )
        content = f"<candidates>\n{items}</candidates>\n\n<question>\n{question}\n</question>"
        key = hashlib.sha256(f"{self.model}|{self.system}|{content}".encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            reply = json.loads(path.read_text())["reply"]
        else:
            self.calls += 1
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=16,
                # SDK 1.x は temperature 引数を持たないため extra_body で送る。
                # 温度指定を受け付けるのは Haiku 4.5 などの旧世代のみ（Haiku 5.5 等は 400）。
                extra_body={"temperature": 0},
                system=self.system,
                messages=[{"role": "user", "content": content}],
            )
            reply = None
            if msg.stop_reason != "refusal":
                reply = "".join(b.text for b in msg.content if b.type == "text")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"reply": reply}, ensure_ascii=False))
        if reply is None:
            return None
        rid = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", reply, flags=re.I)
        return rid if any(c["id"] == rid for c in cands) else None


def decide(question: str, qvec: list[float], faq: list[dict], cfg: dict,
           router: "Router | None") -> tuple[str, str | None, list[str]]:
    """engine.php の Tier0 → Tier1 → Tier1r と同じ順で定型回答を決める。

    戻り値: (tier, 採用したFAQ id, 判定段に渡す候補id)。
      tier は t0 / t1 / t1r / pending（判定段の対象だが --router なし）/ none。
    """
    got0, s0 = best_faq_match_t0(question, faq)
    if s0 >= cfg["tau0"]:
        return "t0", got0, []
    ranked = rank_faq_t1(qvec, faq)
    if ranked and ranked[0][1] >= cfg["tau1"]:
        return "t1", ranked[0][0], []
    cand_ids = [i for i, sc in ranked[: cfg["router_k"]] if sc >= cfg["tau1_cand"]]
    if not cand_ids:
        return "none", None, []
    if router is None:
        return "pending", None, cand_ids
    by_id = {e["id"]: e for e in faq}
    picked = router(question, [by_id[i] for i in cand_ids])
    return ("t1r", picked, cand_ids) if picked else ("none", None, cand_ids)


def light_variants(q: str) -> list[str]:
    """想定問答の質問文と、訪問者が付けがちな軽い前置き・後置き。"""
    return [q, f"あの、{q}", f"{q}教えてください"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def best_faq_match_t0(query: str, faq: list[dict]) -> tuple[str, float]:
    best_id, best_s = "", 0.0
    for e in faq:
        for q in e["questions"]:
            s = bigram_cosine(query, q)
            if s > best_s:
                best_id, best_s = e["id"], s
    return best_id, best_s


def rank_faq_t1(qvec: list[float], faq: list[dict]) -> list[tuple[str, float]]:
    scored = [
        (e["id"], max(cosine(qvec, v) for v in e["qvecs"]))
        for e in faq
    ]
    return sorted(scored, key=lambda x: -x[1])


def sweep(results_pos: list[tuple[bool, float]], results_neg: list[float],
          taus: list[float]) -> list[dict]:
    """閾値ごとの 正答カバー率（top1正解かつスコア≥τ）と 拒否失敗数（負例でスコア≥τ）。"""
    rows = []
    for tau in taus:
        covered = sum(1 for ok, s in results_pos if ok and s >= tau)
        false_ans = sum(1 for s in results_neg if s >= tau)
        rows.append({
            "tau": tau,
            "coverage": covered / len(results_pos),
            "false_answers": false_ans,
        })
    return rows


def recommend(rows: list[dict]) -> float | None:
    """負例の誤回答ゼロでカバー率最大のτ（同率なら低い方＝カバー広い方）。"""
    ok = [r for r in rows if r["false_answers"] == 0]
    if not ok:
        return None
    best = max(ok, key=lambda r: r["coverage"])
    return best["tau"]


def fmt_rows(rows: list[dict], n_neg: int) -> str:
    out = ["| τ | 正答カバー率 | 拒否失敗（誤回答） |", "|---|---|---|"]
    for r in rows:
        out.append(f"| {r['tau']:.2f} | {r['coverage']*100:.0f}% | {r['false_answers']}/{n_neg} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["voyage", "mock"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--kb", default=str(RAG_DIR / "dist/kb.json"))
    ap.add_argument("--router", action="store_true",
                    help="判定段（Tier1r）を実際にAnthropic APIで評価する（要 ANTHROPIC_API_KEY・anthropic SDK）")
    ap.add_argument("--strict", action="store_true",
                    help="想定問答の質問文（＋軽い変形）が1件でも定型回答できなければ終了コード1")
    args = ap.parse_args()

    kb = json.loads(Path(args.kb).read_text(encoding="utf-8"))
    if kb["provider"] != args.provider:
        sys.exit(f"kb.json は provider={kb['provider']} で作られています。--provider を合わせるか再ビルドしてください")
    emb = make_embedder(args.provider, args.model or kb["model"], kb["dim"])
    positives, negatives = load_eval_sets()
    faq, chunks = kb["faq"], kb["chunks"]

    report: list[str] = [
        "# 評価レポート",
        "",
        f"- provider: **{kb['provider']}** / model: {kb['model']} / dim: {kb['dim']}",
        f"- 正答セット {len(positives)}件 / 拒否セット {len(negatives)}件 / "
        f"FAQ {len(faq)}エントリ / チャンク {len(chunks)}件",
    ]
    if kb.get("qvec_input_type") != "query":
        report.append("- **注意: この kb.json は質問文を document モードで埋め込んだ旧形式。"
                      "build_kb.py で再ビルドすること（Tier1 の類似度が不当に低く出る）。**")
    if kb["provider"] == "mock":
        report.append("- **注意: mock埋め込み（文字n-gram）による語彙ベースライン。"
                      "意味的な言い換え性能は Voyage での再実行で測ること。**")

    taus = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30〜0.90

    # ---- Tier0 自己テスト（本来の役割: 定型質問・軽い言い回し違いのヒット） ----
    self_cases: list[tuple[str, str]] = []
    for e in faq:
        for q in e["questions"]:
            self_cases.append((q, e["id"]))                      # 完全一致
            self_cases.append((f"あの、{q}", e["id"]))            # 前置き付き
            self_cases.append((f"{q}教えてください", e["id"]))     # 後置き付き
    self_hits = {tau: 0 for tau in (0.70, 0.80, 0.85)}
    self_top1 = 0
    for q, expect in self_cases:
        got, s = best_faq_match_t0(q, faq)
        if got == expect:
            self_top1 += 1
            for tau in self_hits:
                if s >= tau:
                    self_hits[tau] += 1
    n_self = len(self_cases)
    report += [
        "", "## Tier0 自己テスト: 定型質問＋軽い前置き/後置き", "",
        f"- 対象 {n_self}件（コーパス質問文とその軽微変形）/ top1正解率 {self_top1/n_self*100:.0f}%",
        "| τ | ヒット率（この閾値でTier0即答できる率） |", "|---|---|",
    ] + [f"| {tau:.2f} | {hit/n_self*100:.0f}% |" for tau, hit in sorted(self_hits.items())]

    # ---- パイプライン自己テスト: 想定問答にある質問は必ず定型回答できるか ----
    # 本番の設定（config.php）で engine.php と同じ順に判定し、正しいエントリの
    # 定型回答にならなかったものを「リストにあるのに答えられない質問」として列挙する。
    cfg = load_deployed_config()
    router = Router(cfg["router_model"], cfg["school_name"]) if args.router else None
    pipe_cases = [(v, e["id"]) for e in faq for q in e["questions"] for v in light_variants(q)]
    pipe_vecs = emb.embed([q for q, _ in pipe_cases], "query")
    tiers = {"t0": 0, "t1": 0, "t1r": 0, "pending": 0}
    pipe_fails: list[str] = []
    for (q, expect), v in zip(pipe_cases, pipe_vecs):
        tier, got, cands = decide(q, v, faq, cfg, router)
        if tier == "pending" and expect in cands:
            tiers["pending"] += 1          # 判定段に正解を渡せている（--router で実判定）
        elif tier in ("t0", "t1", "t1r") and got == expect:
            tiers[tier] += 1
        elif got:
            pipe_fails.append(f"- {q} → 期待 {expect} / {tier}で誤答 {got}")
        else:
            pipe_fails.append(f"- {q} → 期待 {expect} / 定型回答なし（候補: {', '.join(cands) or 'なし'}）")
    n_pipe = len(pipe_cases)
    n_ok = tiers["t0"] + tiers["t1"] + tiers["t1r"]
    mode = f"判定段あり（{cfg['router_model']}）" if router else "判定段は未実行（--router なし）"
    report += [
        "", "## パイプライン自己テスト: 想定問答の質問は定型回答できるか", "",
        f"- 本番設定 τ0={cfg['tau0']} / τ1={cfg['tau1']} / 候補下限={cfg['tau1_cand']} / 候補数={cfg['router_k']}（config.php）・{mode}",
        f"- 対象 {n_pipe}件（質問文＋軽い前置き/後置き）",
        f"- 定型回答できた: {n_ok/n_pipe*100:.0f}%（Tier0 {tiers['t0']} / Tier1 {tiers['t1']} / 判定段 {tiers['t1r']}）"
        + (f"・判定段に正解を渡せた {tiers['pending']}件" if not router else "")
        + f" / **できなかった: {len(pipe_fails)}件**",
    ]
    if pipe_fails:
        report += ["", "<details><summary>できなかった質問の一覧</summary>", ""] + pipe_fails + ["", "</details>"]

    # ---- 本番設定での言い換え・拒否セットの成績 ----
    pos_vecs_cfg = emb.embed([q for q, _ in positives], "query")
    neg_vecs_cfg = emb.embed([q for q, _ in negatives], "query")
    pos_ok = pos_wrong = pos_pending = 0
    pos_lines: list[str] = []
    for (q, expect), v in zip(positives, pos_vecs_cfg):
        tier, got, cands = decide(q, v, faq, cfg, router)
        if tier == "pending":
            pos_pending += expect in cands
            continue
        if got == expect:
            pos_ok += 1
        elif got:
            pos_wrong += 1
            pos_lines.append(f"- [誤答] {q} → 期待 {expect} / {tier} で {got}")
    neg_canned: list[str] = []
    neg_to_router = 0
    for (q, cat), v in zip(negatives, neg_vecs_cfg):
        tier, got, cands = decide(q, v, faq, cfg, router)
        neg_to_router += bool(cands)
        if got:
            neg_canned.append(f"- [{cat}] {q} → {tier} で {got} を回答")
    n_pos, n_neg = len(positives), len(negatives)
    report += ["", "## 本番設定での成績（言い換え・拒否セット）", ""]
    if router:
        report += [
            f"- 言い換え: 正しい定型回答 {pos_ok}/{n_pos}（{pos_ok/n_pos*100:.0f}%）/ 誤った定型回答 {pos_wrong}件",
            f"- 拒否セット: 定型回答してしまった **{len(neg_canned)}/{n_neg}件**（判定段に回った {neg_to_router}件）",
            f"- 判定APIの新規呼び出し {router.calls}回（残りはキャッシュ）",
        ]
    else:
        report += [
            f"- 言い換え: 判定なしで正解 {pos_ok}/{n_pos} / 判定段に正解を渡せた {pos_pending}件 / 誤答 {pos_wrong}件",
            f"- 拒否セット: 判定なしで定型回答 {len(neg_canned)}/{n_neg}件 / 判定段に回る {neg_to_router}件",
            "- 判定段の実際の成績は `--router` を付けて測る。",
        ]
    if pos_lines or neg_canned:
        report += [""] + pos_lines + neg_canned

    # ---- Tier0 ----
    pos_t0 = []
    for q, expect in positives:
        got, s = best_faq_match_t0(q, faq)
        pos_t0.append((got == expect, s))
    neg_t0 = [best_faq_match_t0(q, faq)[1] for q, _ in negatives]
    rows0 = sweep(pos_t0, neg_t0, taus)
    tau0 = recommend(rows0)
    top1_t0 = sum(1 for ok, _ in pos_t0 if ok) / len(pos_t0)
    report += [
        "", "## Tier0: バイグラム・ファジーマッチ", "",
        f"- 閾値なしtop1正解率: {top1_t0*100:.0f}%",
        f"- **推奨 τ0 = {tau0}**（拒否セット誤回答ゼロでカバー率最大）" if tau0 is not None
        else "- 誤回答ゼロにできるτがない（要調査）",
        "", fmt_rows(rows0, len(negatives)),
    ]

    # ---- Tier1 ----
    pos_vecs = emb.embed([q for q, _ in positives], "query")
    neg_vecs = emb.embed([q for q, _ in negatives], "query")
    pos_t1, r1 = [], 0
    r3 = 0
    for (q, expect), v in zip(positives, pos_vecs):
        ranked = rank_faq_t1(v, faq)
        ids = [i for i, _ in ranked]
        if ids[0] == expect:
            r1 += 1
        if expect in ids[:3]:
            r3 += 1
        pos_t1.append((ids[0] == expect, ranked[0][1]))
    neg_t1 = [rank_faq_t1(v, faq)[0][1] for v in neg_vecs]
    rows1 = sweep(pos_t1, neg_t1, taus)
    tau1 = recommend(rows1)
    report += [
        "", "## Tier1: 埋め込み類似（FAQ質問文）", "",
        f"- Recall@1: {r1/len(positives)*100:.0f}% / Recall@3: {r3/len(positives)*100:.0f}%",
        f"- **推奨 τ1 = {tau1}**（拒否セット誤回答ゼロでカバー率最大）" if tau1 is not None
        else "- 誤回答ゼロにできるτがない（要調査）",
        "", fmt_rows(rows1, len(negatives)),
    ]

    # ---- Tier2 ----
    def best_chunk(v: list[float]) -> float:
        return max(cosine(v, c["vec"]) for c in chunks)

    pos_t2 = sorted(best_chunk(v) for v in pos_vecs)
    neg_t2 = sorted(best_chunk(v) for v in neg_vecs)

    def pct(xs: list[float], p: float) -> float:
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    report += [
        "", "## Tier2: チャンク検索スコア分布（τ2の当たり付け）", "",
        "| セット | min | p25 | p50 | p75 | max |", "|---|---|---|---|---|---|",
        f"| 正答 | {pos_t2[0]:.2f} | {pct(pos_t2,0.25):.2f} | {pct(pos_t2,0.5):.2f} | {pct(pos_t2,0.75):.2f} | {pos_t2[-1]:.2f} |",
        f"| 拒否 | {neg_t2[0]:.2f} | {pct(neg_t2,0.25):.2f} | {pct(neg_t2,0.5):.2f} | {pct(neg_t2,0.75):.2f} | {neg_t2[-1]:.2f} |",
        "",
        f"- 拒否セットの最大値 {neg_t2[-1]:.2f} より上に τ2 を置くと誤回答ゼロ。",
    ]

    # ---- 拒否セットで誤回答したケースの内訳（デバッグ用） ----
    fails = []
    if tau1 is not None:
        for (q, cat), s in zip(negatives, neg_t1):
            if s >= tau1:
                fails.append(f"- [{cat}] {q} (t1={s:.2f})")
    if fails:
        report += ["", "## 拒否セットで τ1 を超えたケース", ""] + fails

    text = "\n".join(report) + "\n"
    out = RAG_DIR / "eval-report.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"→ {out} に保存")
    if args.strict and (pipe_fails or neg_canned or pos_wrong):
        print(f"STRICT NG: 想定問答の取りこぼし {len(pipe_fails)}件 / "
              f"拒否セットへの定型回答 {len(neg_canned)}件 / 言い換えの誤答 {pos_wrong}件")
        sys.exit(1)


if __name__ == "__main__":
    main()
