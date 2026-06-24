"""
파인튜닝 임베딩 검색 검사 스크립트.

사용법:
  # test set에서 "틀린"(정답이 1위가 아닌) 케이스 10개 자연스럽게 출력
  uv run python inspect_retrieval.py

  # 직접 질문 넣어서 top-10 확인
  uv run python inspect_retrieval.py "조용하고 콘센트 있는 카페"
"""

import sys
import json
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel

BASE_DIR      = Path(__file__).resolve().parents[1]
EMB_DIR       = Path(__file__).resolve().parent
RULE_PATH     = BASE_DIR / "data" / "rule_metadata_merged.json"
TEST_PATH     = EMB_DIR / "data" / "test_query_pos.json"
FINETUNED_DIR = EMB_DIR / "models" / "bge-m3-cafe"

TOP_K      = 10
N_FAIL     = 10   # 출력할 틀린 케이스 수


def load_rules():
    rules = json.load(open(RULE_PATH, encoding="utf-8"))["unique_rules"]
    return [r["title"] for r in rules], [r["description"] for r in rules]


def encode(model, texts):
    v = model.encode(texts, batch_size=64, max_length=512)["dense_vecs"]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    titles, descs = load_rules()
    print(f"[INFO] rule {len(titles)}개 로드. 모델 로드 중...")
    model = BGEM3FlagModel(str(FINETUNED_DIR), use_fp16=True)
    desc_emb = encode(model, descs)

    def retrieve(query, k=TOP_K):
        q = encode(model, [query])[0]
        sims = desc_emb @ q
        order = np.argsort(-sims)[:k]
        return [(titles[j], float(sims[j])) for j in order]

    # ── 모드 1: 직접 질문 ──────────────────────────────────
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\n[질문] {query}\n[모델 top-{TOP_K}]")
        for rank, (t, s) in enumerate(retrieve(query), 1):
            print(f"  {rank:2d}. {t}  ({s:.3f})")
        return

    # ── 모드 2: test set 틀린 케이스 ───────────────────────
    test = json.load(open(TEST_PATH, encoding="utf-8"))
    q_emb = encode(model, [t["query"] for t in test])
    sims  = q_emb @ desc_emb.T
    order = np.argsort(-sims, axis=1)

    fails = []
    for i, item in enumerate(test):
        gold = item["title"]
        ranked = [titles[j] for j in order[i][:TOP_K]]
        if gold != ranked[0]:                      # 1위가 정답이 아니면 "틀림"
            pos = ranked.index(gold) + 1 if gold in ranked else None
            fails.append((item["query"], gold, ranked,
                          [float(sims[i][j]) for j in order[i][:TOP_K]], pos))

    print(f"\n[INFO] test {len(test)}개 중 1위 오답 {len(fails)}개. 그중 {N_FAIL}개 출력:\n")
    for q, gold, ranked, scores, pos in fails[:N_FAIL]:
        tag = f"정답 {pos}위" if pos else "정답 top-10 밖 ❌"
        print("=" * 70)
        print(f"[질문] {q}")
        print(f"[정답] {gold}   →   {tag}")
        print(f"[모델 top-{TOP_K}]")
        for rank, (t, s) in enumerate(zip(ranked, scores), 1):
            mark = " ⭐정답" if t == gold else ""
            print(f"  {rank:2d}. {t}  ({s:.3f}){mark}")
    print("=" * 70)


if __name__ == "__main__":
    main()
