"""질문별 × 매장별 '보낼 리뷰' 선택 (토큰 절약).

규칙:
  - 매장 리뷰 50개 이하 → 전부
  - 50개 초과 → 질문과 코사인 유사도 top-50만

임베딩: text-embedding-3-small. 리뷰 임베딩은 review_emb.npy 캐시.
출력: data/review_selection.json  = {qid: {cafe: [리뷰 local index, ...]}}
       (50 이하라 전부인 매장은 저장 안 함 — 채점 시 없으면 '전부'로 처리)
"""
import json
from pathlib import Path

import numpy as np
from openai import OpenAI

import batch_runner as br
import judge

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CAP = 50
MODEL = "text-embedding-3-small"
EMB_CACHE = HERE / "review_emb.npy"


def embed_all(texts, client, chunk=1000):
    out = []
    for i in range(0, len(texts), chunk):
        r = client.embeddings.create(model=MODEL, input=texts[i:i + chunk])
        out += [d.embedding for d in r.data]
        print(f"  임베딩 {min(i + chunk, len(texts))}/{len(texts)}")
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    client = OpenAI()
    groups = judge.load_reviews()
    cafes = sorted(groups)

    # 전 리뷰 평탄화 + 매장별 슬라이스
    flat, sl = [], {}
    for cafe in cafes:
        s = len(flat)
        flat += groups[cafe]
        sl[cafe] = (s, len(flat))
    print(f"매장 {len(cafes)}곳 / 총 리뷰 {len(flat)}개")

    # 리뷰 임베딩 (캐시)
    if EMB_CACHE.exists() and np.load(EMB_CACHE).shape[0] == len(flat):
        emb = np.load(EMB_CACHE)
        print("리뷰 임베딩 캐시 사용")
    else:
        print("리뷰 임베딩 계산...")
        emb = embed_all(flat, client)
        np.save(EMB_CACHE, emb)

    # 질문 임베딩
    qs = judge.load_questions()
    q_emb = embed_all([q["question"] for q in qs], client)

    # 선택
    selection = {}
    capped = 0
    sent_before = sent_after = 0
    for qi, q in enumerate(qs):
        qv = q_emb[qi]
        per_cafe = {}
        for cafe in cafes:
            s, e = sl[cafe]
            n = e - s
            sent_before += n
            if n <= CAP:
                sent_after += n
                continue  # 전부 → 저장 안 함
            sims = emb[s:e] @ qv
            top = np.argsort(-sims)[:CAP]
            per_cafe[cafe] = sorted(int(i) for i in top)
            capped += 1
            sent_after += CAP
        selection[q["qid"]] = per_cafe

    (DATA / "review_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False), encoding="utf-8")

    print(f"\n캡 적용된 (질문×매장) 쌍: {capped:,} / {len(qs) * len(cafes):,}")
    print(f"보낼 리뷰 총량: {sent_before:,} → {sent_after:,}  ({sent_after/sent_before*100:.0f}%)")
    print(f"저장: {DATA / 'review_selection.json'}")


if __name__ == "__main__":
    main()
