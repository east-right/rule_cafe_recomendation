"""M5 — 추천 품질: 감성분리(현행) 시스템 vs 무작위 기준선, 181 qrels, TOP_K 10/15.

오프라인(OpenAI 0). eval_pipeline 함수를 재사용하되 평가셋은 181 기준(qrels_181.csv).
  - 시스템: BGE-M3 검색 → GGUF 룰선택 → 감성분리 recommend(top-3). TOP_K(후보 룰 수) 10·15 비교.
  - 무작위: item 보유 181곳에서 무작위 top-3 (seed 다수 평균).
  - 전체 50문항 + Rule 선택된 문항 부분집합.
선행: 모델서버(:8001) 내리고 실행(인프로세스 BGE-M3+GGUF 로드 → OOM 방지).
"""
from __future__ import annotations
import json, random, subprocess, sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval_data"))
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import eval_pipeline as ep  # 함수 재사용

QRELS_181 = ROOT / "eval_data" / "data" / "qrels_181.csv"
OUTDIR = ROOT / "eval_data" / "results" / "M5"
SEEDS = 50  # 무작위 baseline 평균 횟수


def metrics_for(ranked, qrel):
    return {
        "hit1": ep.hit_at_k(ranked, qrel, 1),
        "hit3": ep.hit_at_k(ranked, qrel, 3),
        "recall3": ep.recall_at_k(ranked, qrel, 3),
        "ndcg3": ep.ndcg_at_k(ranked, qrel, 3),
        "mrr": ep.mrr(ranked, qrel),
    }


def agg(rows):
    keys = ("hit1", "hit3", "recall3", "ndcg3", "mrr")
    return {k: round(mean(r[k] for r in rows), 4) for k in keys} if rows else {k: 0 for k in keys}


def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rules = ep.load_rules()
    kw_map = ep.load_rule_keywords()
    cafe_items = ep.load_cafe_items()          # 181 (DB)
    questions = ep.load_questions()            # 50
    qrels_df = pd.read_csv(QRELS_181)
    cafes_with_items = sorted(cafe_items.keys())
    print(f"[M5] items={len(cafes_with_items)} questions={len(questions)} qrels_181 rows={len(qrels_df)}")

    print("BGE-M3 로드..."); embedder = SentenceTransformer(ep.EMBED_MODEL)
    rule_emb = ep.embed_rules(embedder, rules)
    print("GGUF 로드..."); llm = Llama(model_path=str(ep.GGUF), n_ctx=4096, verbose=False)

    qrel_map = {qid: dict(zip(g["cafe"], g["grade"]))
                for qid, g in qrels_df.groupby("qid")}

    def run_system(topk):
        ep.TOP_K = topk
        all_rows, sel_rows = [], []
        none_cnt = 0; detail = []
        for q in questions:
            qid, question = q["qid"], q["question"]
            qrel = qrel_map.get(qid, {})
            cands = ep.retrieve(question, embedder, rules, rule_emb)
            raw = ep.select_rule(llm, question, cands)
            title = ep.resolve_title(raw, kw_map, cands)
            if title is None or raw == "none":
                none_cnt += 1; ranked = []
            else:
                ranked, _ = ep.recommend(kw_map[title], cafe_items)
            m = metrics_for(ranked, qrel)
            all_rows.append(m)
            if title is not None and raw != "none":
                sel_rows.append(m)
            detail.append({"qid": qid, "rule": title or f"none({raw})", "top3": ranked, **m})
        return {"topk": topk, "none_count": none_cnt,
                "all": agg(all_rows), "rule_selected": agg(sel_rows),
                "n_all": len(all_rows), "n_rule_selected": len(sel_rows),
                "detail": detail}

    print("시스템 TOP_K=10..."); sys10 = run_system(10)
    print("시스템 TOP_K=15..."); sys15 = run_system(15)

    # 무작위 baseline (item 보유 181곳에서 top-3, SEEDS 평균)
    print(f"무작위 baseline ({SEEDS} seeds)...")
    seed_aggs = []
    for s in range(SEEDS):
        rng = random.Random(s); rows = []
        for q in questions:
            qrel = qrel_map.get(q["qid"], {})
            ranked = rng.sample(cafes_with_items, 3)
            rows.append(metrics_for(ranked, qrel))
        seed_aggs.append(agg(rows))
    rand = {k: round(mean(a[k] for a in seed_aggs), 4) for k in ("hit1", "hit3", "recall3", "ndcg3", "mrr")}

    doc = {
        "task": "M5", "commit": head, "qrels": "eval_data/data/qrels_181.csv",
        "n_questions": len(questions), "n_items": len(cafes_with_items),
        "rel_threshold": ep.REL, "top_n": ep.TOP_N, "random_seeds": SEEDS,
        "note": "오프라인·감성분리 현행 recommend. 시스템은 BGE-M3+GGUF 재추론(결정적). 무작위는 181곳 top-3 평균.",
        "system_topk10": {k: sys10[k] for k in ("none_count", "all", "rule_selected", "n_rule_selected")},
        "system_topk15": {k: sys15[k] for k in ("none_count", "all", "rule_selected", "n_rule_selected")},
        "random_baseline_all": rand,
        "detail_topk10": sys10["detail"],
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "quality.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    def line(name, m): return f"  {name:22s} H@1={m['hit1']:.4f} H@3={m['hit3']:.4f} NDCG@3={m['ndcg3']:.4f} MRR={m['mrr']:.4f} R@3={m['recall3']:.4f}"
    print("\n==== M5 품질 (181 qrels) ====")
    print(f"[TOP_K=10] none={sys10['none_count']}/50  rule_selected={sys10['n_rule_selected']}")
    print(line("시스템 전체(50)", sys10["all"]))
    print(line("시스템 룰선택", sys10["rule_selected"]))
    print(f"[TOP_K=15] none={sys15['none_count']}/50  rule_selected={sys15['n_rule_selected']}")
    print(line("시스템 전체(50)", sys15["all"]))
    print(line("시스템 룰선택", sys15["rule_selected"]))
    print(line("무작위 기준선(50)", rand))
    print(f"\n저장: {OUTDIR/'quality.json'}  (commit {head[:7]})")


if __name__ == "__main__":
    main()
