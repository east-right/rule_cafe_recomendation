import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from opensearchpy import OpenSearch
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

# ── 상수 ──────────────────────────────────────────────────
INDEX_NAME = "cafe_rules"
MODEL_NAME = "east-right/bge-m3-cafe-finetuned"
EMBEDDING_DIM = 1024


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )


def create_index(client: OpenSearch) -> None:
    if client.indices.exists(index=INDEX_NAME):
        print(f"[INFO] 인덱스 '{INDEX_NAME}' 이미 존재함. 삭제 후 재생성합니다.")
        client.indices.delete(index=INDEX_NAME)

    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
            }
        },
        "mappings": {
            "properties": {
                "title": {"type": "keyword"},
                "description": {"type": "text"},
                "description_vector": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                    },
                },
            }
        },
    }

    client.indices.create(index=INDEX_NAME, body=mapping)
    print(f"[INFO] 인덱스 '{INDEX_NAME}' 생성 완료")


def load_rules() -> list[dict]:
    with open(RULE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    rules = data.get("unique_rules", [])
    print(f"[INFO] rule 총 {len(rules)}개 로드 완료")
    return rules


def index_rules(client: OpenSearch, rules: list[dict], model: BGEM3FlagModel) -> None:
    descriptions = [rule["description"] for rule in rules]

    print("[INFO] description 임베딩 중...")
    vecs = model.encode(descriptions, batch_size=16, max_length=512)["dense_vecs"]
    embeddings = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    print("[INFO] OpenSearch에 업로드 중...")
    for i, rule in enumerate(tqdm(rules)):
        doc = {
            "title": rule["title"],
            "description": rule["description"],
            "description_vector": embeddings[i].tolist(),
        }
        client.index(index=INDEX_NAME, id=str(i), body=doc)

    print(f"[INFO] 총 {len(rules)}개 rule 업로드 완료")


def main():
    print("[INFO] BGE-M3 파인튜닝 모델 로드 중...")
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)

    client = get_client()
    create_index(client)
    rules = load_rules()
    index_rules(client, rules, model)

    client.indices.refresh(index=INDEX_NAME)
    count = client.count(index=INDEX_NAME)["count"]
    print(f"[INFO] 인덱스 내 문서 수: {count}")


if __name__ == "__main__":
    main()
