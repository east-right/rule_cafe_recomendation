"""
Brute-force 코사인 vs OpenSearch KNN 검색 방식 비교
test_query_pos.json 기준으로 Recall@K 비교
"""

import json
import os
from pathlib import Path

import numpy as np
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
RULE_PATH = ROOT / "data" / "rule_metadata_merged.json"
QUESTION_PATH = ROOT / "embedding" / "data" / "test_query_pos.json"

# ── 상수 ──────────────────────────────────────────────────
INDEX_NAME = "cafe_rules"
MODEL_NAME = "east-right/bge-m3-cafe-finetuned"
TOP_K_LIST = [1, 5, 10]


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )


def load_data():
    with open(RULE_PATH, encoding="utf-8") as f:
        rule_data = json.load(f)
    with open(QUESTION_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    # 리스트 or 딕셔너리 모두 대응
    if isinstance(questions, dict):
        questions = list(questions.values())

    descriptions = [r["description"] for r in rule_data["unique_rules"]]
    titles = [r["title"] for r in rule_data["unique_rules"]]

    return questions, descriptions, titles


# ── Brute-force 코사인 ─────────────────────────────────────
def evaluate_brute_force(
    model: SentenceTransformer,
    questions: list[dict],
    descriptions: list[str],
    titles: list[str],
    top_k_list: list[int],
) -> dict:
    print("\n[Brute-force] description 임베딩 중...")
    desc_emb = model.encode(descriptions, batch_size=16, normalize_embeddings=True, show_progress_bar=True)

    print("[Brute-force] query 임베딩 중...")
    queries = [q["query"] for q in questions]
    query_emb = model.encode(queries, batch_size=16, normalize_embeddings=True, show_progress_bar=True)

    # 코사인 유사도 행렬 (normalize된 벡터의 내적 = 코사인)
    sim = np.dot(query_emb, desc_emb.T)

    results = {k: 0 for k in top_k_list}
    for i, q in enumerate(questions):
        ranked_titles = [titles[idx] for idx in np.argsort(sim[i])[::-1]]
        for k in top_k_list:
            if q["title"] in ranked_titles[:k]:
                results[k] += 1

    total = len(questions)
    return {k: round(v / total, 4) for k, v in results.items()}


# ── OpenSearch KNN ─────────────────────────────────────────
def evaluate_knn(
    model: SentenceTransformer,
    client: OpenSearch,
    questions: list[dict],
    top_k_list: list[int],
) -> dict:
    max_k = max(top_k_list)
    results = {k: 0 for k in top_k_list}

    print("\n[KNN] OpenSearch 검색 중...")
    for q in tqdm(questions):
        query_vector = model.encode(q["query"], normalize_embeddings=True).tolist()

        body = {
            "size": max_k,
            "query": {
                "knn": {
                    "description_vector": {
                        "vector": query_vector,
                        "k": max_k,
                    }
                }
            },
            "_source": ["title"],
        }

        response = client.search(index=INDEX_NAME, body=body)
        ranked_titles = [hit["_source"]["title"] for hit in response["hits"]["hits"]]

        for k in top_k_list:
            if q["title"] in ranked_titles[:k]:
                results[k] += 1

    total = len(questions)
    return {k: round(v / total, 4) for k, v in results.items()}


def main():
    print("데이터 로드 중...")
    questions, descriptions, titles = load_data()
    print(f"  질문 수: {len(questions)}개 | rule 수: {len(descriptions)}개")

    print("\nBGE 모델 로드 중...")
    model = SentenceTransformer(MODEL_NAME, token=HF_TOKEN)

    client = get_client()

    # 평가
    bf_scores = evaluate_brute_force(model, questions, descriptions, titles, TOP_K_LIST)
    knn_scores = evaluate_knn(model, client, questions, TOP_K_LIST)

    # 결과 출력
    print("\n" + "=" * 55)
    print("Brute-force 코사인 vs OpenSearch KNN 비교")
    print("=" * 55)
    print(f"{'K':<12} {'Brute-force':>14} {'KNN':>10} {'Δ':>10}")
    print("-" * 55)
    for k in TOP_K_LIST:
        bf, knn = bf_scores[k], knn_scores[k]
        delta = round(knn - bf, 4)
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"Recall@{k:<5} {bf:>14} {knn:>10} {arrow}{abs(delta):>9}")
    print("=" * 55)
    print(f"\n총 질문 수: {len(questions)}개")


if __name__ == "__main__":
    main()