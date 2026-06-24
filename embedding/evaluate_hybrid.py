"""
하이브리드 검색(dense + sparse) 효과 측정 — OpenSearch 없이 로컬에서.

- dense : 파인튜닝 모델 (models/bge-m3-cafe)
- sparse: base BAAI/bge-m3 의 lexical weights (sparse 헤드는 dense 파인튜닝 때 미학습이라 base 사용)
- 결합 : (1) min-max 정규화 후 alpha 가중합 스윕  (2) RRF(rank fusion)

Recall@1/5/10 을 dense / sparse / 하이브리드 각각 비교.

사용: uv run python evaluate_hybrid.py
"""

import json
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel

BASE_DIR      = Path(__file__).resolve().parents[1]
EMB_DIR       = Path(__file__).resolve().parent
TEST_PATH     = EMB_DIR / "data" / "test_query_pos.json"
RULE_PATH     = BASE_DIR / "data" / "rule_metadata_merged.json"
FINETUNED_DIR = EMB_DIR / "models" / "bge-m3-cafe"
BASE_MODEL    = "BAAI/bge-m3"
TOP_K         = [1, 5, 10]
RRF_K0        = 60


def load():
    test  = json.load(open(TEST_PATH, encoding="utf-8"))
    rules = json.load(open(RULE_PATH, encoding="utf-8"))["unique_rules"]
    return test, [r["title"] for r in rules], [r["description"] for r in rules]


def enc_dense(model, texts):
    v = model.encode(texts, batch_size=64, max_length=512)["dense_vecs"]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sparse_matrix(model, queries, descs):
    qw = model.encode(queries, return_dense=False, return_sparse=True,
                      return_colbert_vecs=False)["lexical_weights"]
    dw = model.encode(descs, return_dense=False, return_sparse=True,
                      return_colbert_vecs=False)["lexical_weights"]
    M = np.zeros((len(queries), len(descs)), dtype=np.float32)
    for i, q in enumerate(qw):
        for j, d in enumerate(dw):
            M[i, j] = model.compute_lexical_matching_score(q, d)
    return M


def recall(sim, test, titles, ks):
    res = {k: 0 for k in ks}
    for i, item in enumerate(test):
        ranked = [titles[j] for j in np.argsort(-sim[i])]
        for k in ks:
            if item["title"] in ranked[:k]:
                res[k] += 1
    n = len(test)
    return {k: round(res[k] / n, 4) for k in ks}


def minmax(M):
    mn = M.min(axis=1, keepdims=True)
    mx = M.max(axis=1, keepdims=True)
    return (M - mn) / (mx - mn + 1e-9)


def rrf(dense, sparse, titles, test, ks, k0=RRF_K0):
    """reciprocal rank fusion (스케일 무관)"""
    n_q, n_d = dense.shape
    drank = np.argsort(-dense, axis=1)
    srank = np.argsort(-sparse, axis=1)
    score = np.zeros((n_q, n_d))
    for i in range(n_q):
        for r, j in enumerate(drank[i]):
            score[i, j] += 1.0 / (k0 + r)
        for r, j in enumerate(srank[i]):
            score[i, j] += 1.0 / (k0 + r)
    return recall(score, test, titles, ks)


def main():
    test, titles, descs = load()
    queries = [t["query"] for t in test]
    print(f"[INFO] test {len(test)} / rule {len(titles)}")

    print("[INFO] dense (파인튜닝 모델) 인코딩...")
    ft = BGEM3FlagModel(str(FINETUNED_DIR), use_fp16=True)
    d = enc_dense(ft, queries) @ enc_dense(ft, descs).T
    del ft

    print("[INFO] sparse (base bge-m3) 인코딩...")
    base = BGEM3FlagModel(BASE_MODEL, use_fp16=True)
    s = sparse_matrix(base, queries, descs)

    print("\n================= Recall =================")
    print("Dense  :", recall(d, test, titles, TOP_K))
    print("Sparse :", recall(s, test, titles, TOP_K))

    print("\n--- 가중합 (alpha = dense 비중) ---")
    dn, sn = minmax(d), minmax(s)
    best = (None, -1)
    for a in [x / 10 for x in range(11)]:
        r = recall(a * dn + (1 - a) * sn, test, titles, TOP_K)
        print(f"  alpha={a:.1f}: {r}")
        if r[10] > best[1]:
            best = (a, r[10], r)
    print(f"\n  ▶ 최고 @10: alpha={best[0]} → {best[2]}")

    print("\n--- RRF ---")
    print("  RRF:", rrf(d, s, titles, test, TOP_K))


if __name__ == "__main__":
    main()
