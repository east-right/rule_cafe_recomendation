"""
sLLM 파인튜닝 데이터 생성 스크립트

입력: 사용자 질문 + OpenSearch로 검색한 10개 candidate rule (title + description)
출력: 정답 rule title 또는 "none" (정답이 candidates에 없는 경우)

저장 형식: JSONL
{
    "query": "질문",
    "candidates": [
        {"rank": 1, "title": "...", "description": "..."},
        ...
    ],
    "answer": "정답 title" or "none"
}
"""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── 환경변수 로드 ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_READ")

# ── 경로 설정 ──────────────────────────────────────────────
# 이 부분만 수정
QUESTION_PATH = ROOT / "keyword_selection" / "data" / "keyword_questions.jsonl"
OUTPUT_PATH = ROOT / "keyword_selection" / "data" / "finetune_keyword_selection.jsonl"

# ── 상수 ──────────────────────────────────────────────────
INDEX_NAME = "cafe_rules"
MODEL_NAME = "east-right/bge-m3-cafe-finetuned"
TOP_K = 10
SEED = 42


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )


def search_candidates(
    query_vector: list[float],
    client: OpenSearch,
    top_k: int = TOP_K,
) -> list[dict]:
    body = {
        "size": top_k,
        "query": {
            "knn": {
                "description_vector": {
                    "vector": query_vector,
                    "k": top_k,
                }
            }
        },
        "_source": ["title", "description"],
    }

    response = client.search(index=INDEX_NAME, body=body)
    return [
        {
            "title": hit["_source"]["title"],
            "description": hit["_source"]["description"],
        }
        for hit in response["hits"]["hits"]
    ]


def main():
    random.seed(SEED)

    print("데이터 로드 중...")
    with open(QUESTION_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    print(f"  총 질문 수: {len(questions)}개")

    print("\nBGE 모델 로드 중...")
    model = SentenceTransformer(MODEL_NAME, token=HF_TOKEN)

    client = get_client()

    print("\nquery 임베딩 중...")
    queries = [q["query"] for q in questions]
    query_embeddings = model.encode(
        queries,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    count_match = 0
    count_none = 0

    print("\n파인튜닝 데이터 생성 중...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(tqdm(questions)):
            candidates = search_candidates(
                query_embeddings[i].tolist(),
                client,
            )

            candidate_titles = [c["title"] for c in candidates]

            # 정답이 candidates에 있으면 정답 title, 없으면 "none"
            if q["title"] in candidate_titles:
                answer = q["title"]
                count_match += 1
            else:
                answer = "none"
                count_none += 1

            # 위치 편향 방지: 랜덤 셔플 후 rank 재부여
            random.shuffle(candidates)
            for rank, c in enumerate(candidates, 1):
                c["rank"] = rank

            record = {
                "query": q["query"],
                "candidates": candidates,
                "answer": answer,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n완료!")
    print(f"  정답 있는 데이터: {count_match}개")
    print(f"  정답 없는 데이터 (none): {count_none}개")
    print(f"  총 생성: {count_match + count_none}개")
    print(f"  저장 경로: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()