import os
from pathlib import Path

from dotenv import load_dotenv
from opensearchpy import OpenSearch
# sentence_transformers(torch)는 load_model에서 lazy import.
# get_client/INDEX_NAME만 쓰는 app(모델서버에 임베딩 위임)이 torch를 안 끌게 하기 위함.

# ── 환경변수 로드 ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_READ")

# ── 상수 ──────────────────────────────────────────────────
INDEX_NAME = "cafe_rules"
MODEL_NAME = "east-right/bge-m3-cafe-finetuned"
TOP_K = 10


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )


def load_model():
    from sentence_transformers import SentenceTransformer  # lazy: torch는 여기서만
    return SentenceTransformer(MODEL_NAME, token=HF_TOKEN)


def search(
    query: str,
    client: OpenSearch,
    model,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    쿼리를 임베딩해서 OpenSearch에서 유사한 rule top_k개 반환

    Returns:
        [{"title": ..., "description": ..., "score": ...}, ...]
    """
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

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
    hits = response["hits"]["hits"]

    results = [
        {
            "title": hit["_source"]["title"],
            "description": hit["_source"]["description"],
            "score": hit["_score"],
        }
        for hit in hits
    ]

    return results


if __name__ == "__main__":
    # 테스트
    model = load_model()
    client = get_client()

    test_queries = [
        "조용하고 콘센트 있는 카페 추천해줘",
        "뷰가 좋은 카페 어디 있어?",
        "디저트 맛있는 카페 알려줘",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"쿼리: {query}")
        print(f"{'='*60}")
        results = search(query, client, model)
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r['title']} (score: {r['score']:.4f})")
            print(f"     {r['description'][:50]}...")