"""룰기반 vs ML기반(LambdaMART) 추천 비교 — 동일 골드 qrels·동일 지표.

두 시스템을 같은 50개 질문 / 같은 골드 qrels(eval_data/data/qrels.csv)로 평가하고,
질문별 추천 매장 + 점수 + 골드 관련도를 CSV로 저장한다.

[룰기반]  eval_pipeline.py 재사용:
   BGE-M3 코사인 top-10 → GGUF 룰선택 → soar 트레이스(operator→neg→tiebreak) top-N

[ML기반]  LambdaMART (LightGBM lambdarank):
   피처(질문↔프로필 코사인, 질문↔긍정키워드 코사인, 카페 사전확률) → 랭킹 top-N
   - 누수 방지: GroupKFold(질문 단위) out-of-fold 예측
   - 새 질문은 자유텍스트(키워드 라벨 없음) → 키워드 라벨 불필요한 피처만 사용
   - 라벨 = 골드 qrels grade (LLM-judge, 랭커와 독립 → 순환 아님)

산출:
   eval_data/data/compare_recommendations.csv   (질문별 top-5 × 두 시스템, long format)
   eval_data/data/compare_summary.csv           (시스템별 지표 요약)

사용:
   uv run python eval_data/eval_compare.py
   uv run python eval_data/eval_compare.py --dry 5
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import eval_pipeline as ep  # 룰 파이프라인 함수·지표 재사용

ROOT = ep.ROOT
DATA = ep.DATA
CAFE_DB = ep.CAFE_DB

TOP_N = 5          # CSV에 담을 질문별 추천 매장 수
REL = ep.REL       # grade >= REL = 관련(적중)
FEATS = ["emb_profile", "emb_poskw", "n_pos", "n_neg", "polarity", "n_menu"]


# ── ML: 카페 아이템 데이터 (노트북 build_profile 재현) ──────────

def build_items(embedder) -> pd.DataFrame:
    conn = sqlite3.connect(str(CAFE_DB))
    cafes = pd.read_sql("SELECT name, review_summary FROM cafes", conn)
    kw = pd.read_sql("SELECT cafe_name, sentiment, keyword FROM cafe_keywords", conn)
    menus = pd.read_sql("SELECT DISTINCT cafe_name, menu_name FROM cafe_menus", conn)
    conn.close()

    pos = kw[kw.sentiment == "긍정"].groupby("cafe_name")["keyword"].apply(list)
    neg = kw[kw.sentiment == "부정"].groupby("cafe_name")["keyword"].apply(list)
    mn = menus.groupby("cafe_name")["menu_name"].apply(list)

    # 랭킹 유니버스 = 키워드 보유 카페 (룰 시스템과 동일 범위)
    universe = sorted(kw.cafe_name.unique())
    rs_map = dict(zip(cafes.name, cafes.review_summary))

    rows = []
    for name in universe:
        p = pos.get(name, [])
        n = neg.get(name, [])
        m = mn.get(name, [])
        rs = rs_map.get(name, "")
        parts = []
        if p:
            parts.append("특징: " + ", ".join(p))
        if m:
            parts.append("메뉴: " + ", ".join(m[:10]))
        if isinstance(rs, str) and rs.strip():
            parts.append("리뷰: " + rs)
        rows.append({
            "cafe": name, "pos": p, "neg": n, "menus": m,
            "profile_text": " | ".join(parts),
            "poskw_text": ", ".join(p) if p else "",
        })
    items = pd.DataFrame(rows)

    print(f"  카페 아이템 {len(items)}개 임베딩 중...")
    items["emb_profile"] = list(embedder.encode(
        items["profile_text"].tolist(), normalize_embeddings=True, show_progress_bar=False))
    # 긍정키워드 임베딩(빈 문자열은 0 벡터)
    pk_emb = embedder.encode(
        [t if t else "없음" for t in items["poskw_text"]],
        normalize_embeddings=True, show_progress_bar=False)
    for i, t in enumerate(items["poskw_text"]):
        if not t:
            pk_emb[i] = 0.0
    items["emb_poskw"] = list(pk_emb)
    return items


def build_feature_table(questions, items, qrels_df, embedder) -> pd.DataFrame:
    """(질문 × 카페) 피처 + 골드 grade 라벨 학습표."""
    q_emb = {q["qid"]: embedder.encode([q["question"]], normalize_embeddings=True)[0]
             for q in questions}
    prof = np.vstack(items["emb_profile"].to_numpy())
    posk = np.vstack(items["emb_poskw"].to_numpy())
    n_pos = items["pos"].apply(len).to_numpy()
    n_neg = items["neg"].apply(len).to_numpy()
    n_menu = items["menus"].apply(len).to_numpy()
    polarity = n_pos / (n_pos + n_neg + 1)
    cafes = items["cafe"].tolist()

    # qrels 조회용
    grade_map = {(r.qid, r.cafe): r.grade for r in qrels_df.itertuples()}

    rows = []
    for q in questions:
        qid = q["qid"]
        qv = q_emb[qid].astype(np.float32)
        f_prof = prof @ qv
        f_pos = posk @ qv
        for i, cafe in enumerate(cafes):
            rows.append({
                "qid": qid, "cafe": cafe,
                "emb_profile": float(f_prof[i]), "emb_poskw": float(f_pos[i]),
                "n_pos": int(n_pos[i]), "n_neg": int(n_neg[i]),
                "polarity": float(polarity[i]), "n_menu": int(n_menu[i]),
                "grade": int(grade_map.get((qid, cafe), 0)),
            })
    return pd.DataFrame(rows)


def train_oof(table: pd.DataFrame) -> pd.DataFrame:
    """GroupKFold(질문 단위) out-of-fold LambdaMART 예측 → table에 ml_score 추가."""
    qids = table["qid"].to_numpy()
    uniq = table["qid"].unique()
    n_splits = min(5, len(uniq))
    table = table.copy()
    table["ml_score"] = np.nan

    gkf = GroupKFold(n_splits=n_splits)
    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 3, 5],
        "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20, "verbose": -1,
    }
    for tr_idx, te_idx in gkf.split(table, groups=qids):
        tr = table.iloc[tr_idx].sort_values("qid")
        te = table.iloc[te_idx]
        g_tr = tr.groupby("qid", sort=True).size().to_numpy()
        dset = lgb.Dataset(tr[FEATS].to_numpy(), label=tr["grade"].to_numpy(), group=g_tr)
        ranker = lgb.train(params, dset, num_boost_round=200)
        table.loc[te.index, "ml_score"] = ranker.predict(te[FEATS].to_numpy())
    return table


# ── 지표 (룰·ML 공통) ───────────────────────────────────────

def eval_ranked(ranked: list[str], qrel: dict[str, int]) -> dict:
    return {
        "hit1": ep.hit_at_k(ranked, qrel, 1),
        "hit3": ep.hit_at_k(ranked, qrel, 3),
        "recall3": ep.recall_at_k(ranked, qrel, 3),
        "ndcg3": ep.ndcg_at_k(ranked, qrel, 3),
        "mrr": ep.mrr(ranked, qrel),
    }


# ── 메인 ────────────────────────────────────────────────────

def main(dry: int | None):
    questions = ep.load_questions()
    qrels_df = ep.load_qrels()
    if dry:
        questions = questions[:dry]
        print(f"[dry] 질문 {len(questions)}개만\n")

    print(f"BGE-M3 로드: {ep.EMBED_MODEL}")
    embedder = SentenceTransformer(ep.EMBED_MODEL)

    # ===== ML 기반 =====
    print("\n[ML] 아이템·피처·LambdaMART(GroupKFold OOF)")
    items = build_items(embedder)
    table = build_feature_table(questions, items, qrels_df, embedder)
    table = train_oof(table)

    ml_rank = {}
    for qid, g in table.groupby("qid"):
        top = g.sort_values("ml_score", ascending=False).head(TOP_N)
        ml_rank[qid] = list(zip(top["cafe"], top["ml_score"], top["grade"]))

    # ===== 룰 기반 (eval_pipeline 재사용) =====
    print("\n[룰] BGE 코사인 → GGUF 룰선택 → soar 트레이스")
    rules = ep.load_rules()
    kw_map = ep.load_rule_keywords()
    cafe_items = ep.load_cafe_items()
    rule_emb = ep.embed_rules(embedder, rules)
    llm = Llama(model_path=str(ep.GGUF), n_ctx=4096, verbose=False)
    ep.TOP_N = TOP_N  # 룰도 top-5 뽑도록

    rule_rank = {}
    for q in questions:
        cands = ep.retrieve(q["question"], embedder, rules, rule_emb)
        raw = ep.select_rule(llm, q["question"], cands)
        title = ep.resolve_title(raw, kw_map, cands)
        if title is None or raw == "none":
            rule_rank[q["qid"]] = ("none", [])
        else:
            ranked, detail = ep.recommend(kw_map[title], cafe_items)
            rule_rank[q["qid"]] = (title, [(d["cafe"], d["score"]) for d in detail])

    # ===== CSV + 지표 =====
    long_rows = []
    rule_m, ml_m = [], []
    print(f"\n{'질문':32} | 룰 top1 (grade) | ML top1 (grade)")
    print("-" * 90)
    for q in questions:
        qid, question = q["qid"], q["question"]
        qdf = qrels_df[qrels_df["qid"] == qid]
        qrel = dict(zip(qdf["cafe"], qdf["grade"]))

        title, rlist = rule_rank[qid]
        rule_ranked = [c for c, _ in rlist]
        for rk, (cafe, sc) in enumerate(rlist, 1):
            long_rows.append({
                "qid": qid, "question": question, "system": "rule", "rank": rk,
                "cafe": cafe, "score": round(float(sc), 4),
                "gold_grade": int(qrel.get(cafe, 0)), "selected_rule": title,
            })
        ml_list = ml_rank[qid]
        ml_ranked = [c for c, _, _ in ml_list]
        for rk, (cafe, sc, gr) in enumerate(ml_list, 1):
            long_rows.append({
                "qid": qid, "question": question, "system": "ml", "rank": rk,
                "cafe": cafe, "score": round(float(sc), 4),
                "gold_grade": int(gr), "selected_rule": "",
            })

        rule_m.append(eval_ranked(rule_ranked, qrel))
        ml_m.append(eval_ranked(ml_ranked, qrel))

        rg = qrel.get(rule_ranked[0], 0) if rule_ranked else "-"
        mg = qrel.get(ml_ranked[0], 0) if ml_ranked else "-"
        r1 = rule_ranked[0][:14] if rule_ranked else "(없음)"
        m1 = ml_ranked[0][:14] if ml_ranked else "(없음)"
        print(f"{question[:30]:32} | {r1:14}({rg}) | {m1:14}({mg})")

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(DATA / "compare_recommendations.csv", index=False, encoding="utf-8-sig")

    def agg(ms):
        return {k: round(float(np.mean([m[k] for m in ms])), 4) for k in ms[0]}
    rule_s, ml_s = agg(rule_m), agg(ml_m)
    summary = pd.DataFrame([{"system": "rule", **rule_s}, {"system": "ml", **ml_s}])
    summary.to_csv(DATA / "compare_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'='*62}")
    print(f"  시스템 비교 (질문 {len(questions)}개, 관련기준 grade≥{REL})")
    print(f"{'='*62}")
    print(f"  {'system':6} {'Hit@1':>8} {'Hit@3':>8} {'Recall@3':>10} {'NDCG@3':>8} {'MRR':>8}")
    for name, s in (("rule", rule_s), ("ml", ml_s)):
        print(f"  {name:6} {s['hit1']:>8.4f} {s['hit3']:>8.4f} {s['recall3']:>10.4f} "
              f"{s['ndcg3']:>8.4f} {s['mrr']:>8.4f}")
    print(f"{'='*62}")
    print(f"  CSV: {DATA / 'compare_recommendations.csv'}")
    print(f"       {DATA / 'compare_summary.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", type=int, default=None)
    args = ap.parse_args()
    main(dry=args.dry)
