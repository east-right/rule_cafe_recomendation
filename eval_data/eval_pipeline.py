"""룰선택 + 룰기반 추천 파이프라인 검증 (qrels 대비).

실제 서비스 파이프라인(service/nodes)을 서버·OpenSearch 없이 CPU 자립형으로 재현:
  [1] 검색   : BGE-M3 파인튜닝으로 룰 description 임베딩 → 코사인 top-10
               (OpenSearch KNN과 동일: 정규화 임베딩 + cosine)
  [2] 룰선택 : GGUF Qwen2.5-1.5B (inference/prompt.py와 동일 프롬프트)
  [3] 룰기반 : soar_recommend 트레이스(operator→negative→tiebreak)를 재현,
               top-3 랭킹으로 확장. (impasse LLM 해소·rule_fallback은 제외 — 룰 로직만 평가)

qrels(eval_data/data/qrels.csv) 대비 Hit@1 / Hit@3 / Recall@3 / NDCG@3 / MRR.

사용:
  uv run python eval_data/eval_pipeline.py            # 50개 전체
  uv run python eval_data/eval_pipeline.py --dry 5    # 5개만 (빠른 점검)
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

GGUF = ROOT / "keyword_selection" / "qwen1.5b-cafe-q8_0.gguf"
EMBED_MODEL = "east-right/bge-m3-cafe-finetuned"
RULE_META = ROOT / "data" / "rule_metadata_merged.json"
RULE_KW = ROOT / "data" / "soar_rule_keywords.json"
CAFE_DB = ROOT / "data" / "cafe.db"
QRELS = DATA / "qrels.csv"
QUESTIONS = DATA / "questions_eval.json"
EMB_CACHE = HERE / "rule_emb.npy"

TOP_K = 10       # 후보 룰 수 (실제 파이프라인 TOP_K와 동일)
TOP_N = 3        # 추천 매장 수 (원래 1~2 → 3개로 확장)
REL = 2          # qrels grade >= REL 이면 '관련 있음(추천 적중)'

# inference/prompt.py 와 동일
RULE_SELECT_SYSTEM = (
    "You are a cafe recommendation rule selector.\n"
    "Given a user query and a list of candidate rules, return the title of the most appropriate rule.\n"
    'If no rule is appropriate, return "none".\n'
    "Return only the title. Do not include any explanation."
)


def build_rule_select_user(query: str, candidates: list[dict]) -> str:
    cand = "\n".join(f"{c['rank']}. {c['title']} - {c['description']}" for c in candidates)
    return f"[질문]\n{query}\n\n[후보]\n{cand}"


# ── 데이터 로드 ──────────────────────────────────────────────

def load_rules() -> list[dict]:
    data = json.loads(RULE_META.read_text(encoding="utf-8"))
    return data["unique_rules"]


def load_rule_keywords() -> dict[str, dict]:
    """title → {operator, tiebreak, negative}. OpenSearch에 인덱싱되는 것과 동일 소스."""
    items = json.loads(RULE_KW.read_text(encoding="utf-8"))
    return {
        r["title"]: {
            "operator": r.get("operator_keywords", []),
            "tiebreak": r.get("tiebreak_keywords", []),
            "negative": r.get("negative_keywords", []),
        }
        for r in items
    }


def load_cafe_items() -> dict[str, list[str]]:
    """soar_recommend._load_cafe_items 재현: 매장별 키워드 전부(긍/부정 무관) 평탄 리스트."""
    conn = sqlite3.connect(str(CAFE_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT cafe_name, keyword FROM cafe_keywords").fetchall()
    conn.close()
    items: dict[str, list[str]] = {}
    for r in rows:
        items.setdefault(r["cafe_name"], []).append(r["keyword"])
    return items


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def load_qrels() -> pd.DataFrame:
    return pd.read_csv(QRELS)


# ── [1] 검색 (BGE-M3 코사인 KNN, OpenSearch 대체) ────────────

def embed_rules(embedder, rules: list[dict]) -> np.ndarray:
    if EMB_CACHE.exists():
        cached = np.load(EMB_CACHE)
        if cached.shape[0] == len(rules):
            print("룰 임베딩 캐시 사용")
            return cached
    print("룰 description 임베딩 중...")
    descs = [r["description"] for r in rules]
    emb = embedder.encode(descs, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(EMB_CACHE, emb)
    return emb


def retrieve(question: str, embedder, rules, rule_emb) -> list[dict]:
    qv = embedder.encode([question], normalize_embeddings=True)[0]
    sims = rule_emb @ np.asarray(qv, dtype=np.float32)
    top = np.argsort(-sims)[:TOP_K]
    return [
        {"rank": i + 1, "title": rules[idx]["title"], "description": rules[idx]["description"]}
        for i, idx in enumerate(top)
    ]


# ── [2] 룰선택 (GGUF) ────────────────────────────────────────

def select_rule(llm: Llama, question: str, candidates: list[dict]) -> str:
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": RULE_SELECT_SYSTEM},
            {"role": "user", "content": build_rule_select_user(question, candidates)},
        ],
        max_tokens=32,
        temperature=0.0,
        repeat_penalty=1.1,
    )
    return resp["choices"][0]["message"]["content"].strip()


def resolve_title(raw: str, kw_map: dict, candidates: list[dict]) -> str | None:
    """GGUF 출력 → 실제 룰 title. 정확히 맞으면 그것, 아니면 후보 중 포함된 것, 없으면 None."""
    if raw in kw_map:
        return raw
    for c in candidates:
        if c["title"] in raw:
            return c["title"]
    return None


# ── [3] 룰기반 추천 (soar 트레이스 재현 + top-3 랭킹) ──────────

def recommend(kw: dict, cafe_items: dict[str, list[str]]) -> tuple[list[str], list[dict]]:
    """operator 후보 → negative 감점 → tiebreak 가점 순으로 랭킹, top-3 반환.

    soar_recommend 의 좁혀가기(operator→negative→tiebreak)와 순서가 일치:
      - operator 키워드 매칭 = 후보 자격(전제)
      - negative 키워드 보유 = 감점 (원래 트레이스의 negative narrowing)
      - tiebreak 키워드 보유 수 = 가점 (원래 트레이스의 tiebreak refine)
    점수 동률이면 operator 매칭 수로 보조 정렬.
    """
    op = set(kw["operator"])
    tb = set(kw["tiebreak"])
    neg = set(kw["negative"])

    scored = []
    for cafe, kws in cafe_items.items():
        ks = set(kws)
        op_hit = len(op & ks)
        if op_hit == 0:
            continue  # operator 매칭 없으면 후보 아님
        tb_hit = len(tb & ks)
        neg_hit = len(neg & ks)
        score = tb_hit - neg_hit
        scored.append((score, op_hit, tb_hit, neg_hit, cafe))

    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    ranked = [s[4] for s in scored[:TOP_N]]
    detail = [
        {"cafe": s[4], "score": s[0], "op": s[1], "tb": s[2], "neg": s[3]}
        for s in scored[:TOP_N]
    ]
    return ranked, detail


# ── 지표 ────────────────────────────────────────────────────

def ndcg_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    dcg = sum((2 ** qrel.get(c, 0) - 1) / math.log2(i + 2) for i, c in enumerate(ranked[:k]))
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked: list[str], qrel: dict[str, int]) -> float:
    for i, c in enumerate(ranked):
        if qrel.get(c, 0) >= REL:
            return 1 / (i + 1)
    return 0.0


def hit_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    return float(any(qrel.get(c, 0) >= REL for c in ranked[:k]))


def recall_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    rel_total = sum(1 for g in qrel.values() if g >= REL)
    if rel_total == 0:
        return 0.0
    hit = sum(1 for c in ranked[:k] if qrel.get(c, 0) >= REL)
    return hit / rel_total


# ── 메인 ────────────────────────────────────────────────────

def main(dry: int | None):
    rules = load_rules()
    kw_map = load_rule_keywords()
    cafe_items = load_cafe_items()
    questions = load_questions()
    qrels_df = load_qrels()

    if dry:
        questions = questions[:dry]
        print(f"[dry] 질문 {len(questions)}개만 실행\n")

    print(f"BGE-M3 임베딩 모델 로드: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    rule_emb = embed_rules(embedder, rules)

    print(f"GGUF 로드: {GGUF.name}")
    llm = Llama(model_path=str(GGUF), n_ctx=4096, verbose=False)

    metrics = {"hit1": [], "hit3": [], "recall3": [], "ndcg3": [], "mrr": []}
    none_cnt = 0
    empty_cnt = 0
    results = []

    print("\n검증 시작...\n")
    for q in questions:
        qid, question = q["qid"], q["question"]
        qdf = qrels_df[qrels_df["qid"] == qid]
        qrel = dict(zip(qdf["cafe"], qdf["grade"]))

        candidates = retrieve(question, embedder, rules, rule_emb)
        raw = select_rule(llm, question, candidates)
        title = resolve_title(raw, kw_map, candidates)

        if title is None or raw == "none":
            none_cnt += 1
            ranked, detail = [], []
        else:
            ranked, detail = recommend(kw_map[title], cafe_items)
            if not ranked:
                empty_cnt += 1

        h1 = hit_at_k(ranked, qrel, 1)
        h3 = hit_at_k(ranked, qrel, 3)
        r3 = recall_at_k(ranked, qrel, 3)
        n3 = ndcg_at_k(ranked, qrel, 3)
        m = mrr(ranked, qrel)
        metrics["hit1"].append(h1)
        metrics["hit3"].append(h3)
        metrics["recall3"].append(r3)
        metrics["ndcg3"].append(n3)
        metrics["mrr"].append(m)

        grades = [f"{c}({qrel.get(c, 0)})" for c in ranked]
        results.append({
            "qid": qid, "question": question,
            "selected_rule": title if title else f"none(raw={raw})",
            "top3": ranked,
            "top3_grades": [qrel.get(c, 0) for c in ranked],
            "detail": detail,
            "hit1": h1, "hit3": h3, "recall3": round(r3, 4),
            "ndcg3": round(n3, 4), "mrr": round(m, 4),
        })
        print(f"  [{qid}] 룰={str(title):10s} NDCG@3={n3:.3f} H@1={h1:.0f} | {question[:28]}")
        print(f"        top3: {', '.join(grades) if grades else '(없음)'}")

    n = len(questions)
    print(f"\n{'='*62}")
    print(f"  룰선택 + 룰기반 추천 파이프라인 (질문 {n}개, 관련기준 grade≥{REL})")
    print(f"{'='*62}")
    print(f"  {'Hit@1':>8} {'Hit@3':>8} {'Recall@3':>10} {'NDCG@3':>8} {'MRR':>8}")
    print(f"  {np.mean(metrics['hit1']):>8.4f} {np.mean(metrics['hit3']):>8.4f} "
          f"{np.mean(metrics['recall3']):>10.4f} {np.mean(metrics['ndcg3']):>8.4f} "
          f"{np.mean(metrics['mrr']):>8.4f}")
    print(f"{'='*62}")
    print(f"  none 선택(룰 없음): {none_cnt}/{n}   operator 매칭 0(추천 실패): {empty_cnt}/{n}")

    out = DATA / "pipeline_eval_results.json"
    out.write_text(json.dumps({
        "n_questions": n,
        "rel_threshold": REL,
        "summary": {k: round(float(np.mean(v)), 4) for k, v in metrics.items()},
        "none_count": none_cnt,
        "empty_count": empty_cnt,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  상세 결과: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", type=int, default=None, help="질문 N개만 실행")
    args = ap.parse_args()
    main(dry=args.dry)
