"""
리랭커 효과 측정 — 임베딩 top-20 → cross-encoder 재정렬 → top-10.

- retrieval: 파인튜닝 임베딩(models/bge-m3-cafe)으로 top-20
- rerank   : BAAI/bge-reranker-v2-m3 로 (query, description) 점수 재계산 후 재정렬
- 비교     : 임베딩 단독 Recall vs 리랭킹 후 Recall

리랭킹 후 R@k(k<=20)의 상한 = 임베딩 R@20. 그 천장에 얼마나 근접하는지 본다.

사용: uv run python evaluate_rerank.py
"""

import json
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel, FlagReranker

BASE_DIR      = Path(__file__).resolve().parents[1]
EMB_DIR       = Path(__file__).resolve().parent
TEST_PATH     = EMB_DIR / "data" / "test_query_pos.json"
RULE_PATH     = BASE_DIR / "data" / "rule_metadata_merged.json"
FINETUNED_DIR = EMB_DIR / "models" / "bge-m3-cafe"
RERANK_MODEL  = "BAAI/bge-reranker-v2-m3"

RETRIEVE_N = 20
TOP_K      = [1, 5, 10]


def load():
    test  = json.load(open(TEST_PATH, encoding="utf-8"))
    rules = json.load(open(RULE_PATH, encoding="utf-8"))["unique_rules"]
    return test, [r["title"] for r in rules], [r["description"] for r in rules]


def enc(model, texts):
    v = model.encode(texts, batch_size=64, max_length=512)["dense_vecs"]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def recall(orders, test, titles, ks):
    res = {k: 0 for k in ks}
    for i, item in enumerate(test):
        ranked = [titles[j] for j in orders[i]]
        for k in ks:
            if item["title"] in ranked[:k]:
                res[k] += 1
    n = len(test)
    return {k: round(res[k] / n, 4) for k in ks}


def main():
    test, titles, descs = load()
    queries = [t["query"] for t in test]
    nq = len(queries)
    print(f"[INFO] test {nq} / rule {len(titles)}")

    print("[INFO] 임베딩 검색 (top-20)...")
    ft = BGEM3FlagModel(str(FINETUNED_DIR), use_fp16=True)
    sim = enc(ft, queries) @ enc(ft, descs).T
    del ft
    emb_order = np.argsort(-sim, axis=1)
    topN = emb_order[:, :RETRIEVE_N]

    print("[INFO] 리랭커 로드 + 재정렬...")
    rr = FlagReranker(RERANK_MODEL, use_fp16=True)
    flat = [[queries[i], descs[j]] for i in range(nq) for j in topN[i]]
    scores = np.array(rr.compute_score(flat, batch_size=256, normalize=True)).reshape(nq, RETRIEVE_N)
    reranked = [topN[i][np.argsort(-scores[i])] for i in range(nq)]

    print("\n================= Recall =================")
    print("임베딩 단독   :", recall(emb_order, test, titles, [1, 5, 10, 20]))
    print("리랭킹(top20) :", recall(reranked,  test, titles, TOP_K))
    print("(리랭킹 R@k 상한 = 임베딩 R@20)")


if __name__ == "__main__":
    main()
