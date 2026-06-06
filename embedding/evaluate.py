"""
Recall@K 평가 스크립트
BGE-M3-ko vs 파인튜닝된 BGE-M3 비교
test set으로만 평가
"""

import json
import numpy as np
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parents[1]
EMB_DIR         = Path(__file__).resolve().parent
TEST_PATH       = EMB_DIR / "data" / "test_query_pos.json"   # test set
RULE_PATH       = BASE_DIR / "data" / "rule_metadata_merged.json"
FINETUNED_DIR   = EMB_DIR / "models" / "bge-m3-cafe"

# ── 설정 ──────────────────────────────────────────────────────
BATCH_SIZE  = 32
TOP_K       = [1, 5, 10]
BGE_M3_KO   = "dragonkue/BGE-m3-ko"
# ──────────────────────────────────────────────────────────────


def load_data():
    with open(TEST_PATH, encoding="utf-8") as f:
        query_pos = json.load(f)
    with open(RULE_PATH, encoding="utf-8") as f:
        rule_data = json.load(f)
    descriptions = [r["description"] for r in rule_data["unique_rules"]]
    titles       = [r["title"]       for r in rule_data["unique_rules"]]
    return query_pos, descriptions, titles


def encode_all(model, texts, batch_size=32):
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb   = model.encode(batch, batch_size=batch_size, max_length=512)["dense_vecs"]
        all_emb.append(emb)
    emb = np.vstack(all_emb)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def evaluate(model, query_pos, descriptions, titles, top_k_list):
    print("  description 임베딩 중...")
    desc_emb  = encode_all(model, descriptions, BATCH_SIZE)

    print("  query 임베딩 중...")
    queries   = [p["query"] for p in query_pos]
    query_emb = encode_all(model, queries, BATCH_SIZE)

    sim = np.dot(query_emb, desc_emb.T)

    results = {k: 0 for k in top_k_list}
    for i, pair in enumerate(query_pos):
        ranked = [titles[idx] for idx in np.argsort(sim[i])[::-1]]
        for k in top_k_list:
            if pair["title"] in ranked[:k]:
                results[k] += 1

    total = len(query_pos)
    return {k: round(v / total, 4) for k, v in results.items()}


def main():
    print("📂 데이터 로드 중...")
    query_pos, descriptions, titles = load_data()
    print(f"  test set: {len(query_pos)}개 | descriptions: {len(descriptions)}개")

    # BGE-M3-ko 평가
    print(f"\n📊 [BGE-M3-ko] {BGE_M3_KO} 평가 중...")
    model_ko  = BGEM3FlagModel(BGE_M3_KO, use_fp16=True)
    ko_scores = evaluate(model_ko, query_pos, descriptions, titles, TOP_K)
    print(f"  결과: {ko_scores}")
    del model_ko

    # 파인튜닝 모델 평가
    print(f"\n📊 [Finetuned] 파인튜닝 모델 평가 중...")
    model_ft  = BGEM3FlagModel(str(FINETUNED_DIR), use_fp16=True)
    ft_scores = evaluate(model_ft, query_pos, descriptions, titles, TOP_K)
    print(f"  결과: {ft_scores}")

    # 비교 출력
    print("\n" + "="*50)
    print("📈 BGE-M3-ko vs 파인튜닝 모델 비교 (test set)")
    print("="*50)
    print(f"{'K':<12} {'BGE-M3-ko':>12} {'Finetuned':>12} {'Δ':>10}")
    print("-"*50)
    for k in TOP_K:
        b, a  = ko_scores[k], ft_scores[k]
        delta = round(a - b, 4)
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"Recall@{k:<5} {b:>12} {a:>12} {arrow}{abs(delta):>9}")
    print("="*50)


if __name__ == "__main__":
    main()
