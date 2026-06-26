import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from langfuse import get_client as langfuse_client, observe

from keyword_selection.search import INDEX_NAME, get_client
from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# 모델 추론은 별도 서버(inference.server)가 담당. OpenSearch 검색만 여기서.
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8001")
TOP_K = 10

_os_client = None
_http: httpx.Client | None = None


def _get_os():
    global _os_client
    if _os_client is None:
        _os_client = get_client()
    return _os_client


def _get_http() -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(base_url=MODEL_SERVER_URL, timeout=60.0)
    return _http


def _embed(text: str) -> list[float]:
    resp = _get_http().post("/embed", json={"texts": [text]})
    resp.raise_for_status()
    return resp.json()["vectors"][0]


def _retrieve(question: str) -> list[dict]:
    """질문 임베딩(모델 서버) → OpenSearch KNN top-K."""
    vector = _embed(question)
    res = _get_os().search(
        index=INDEX_NAME,
        body={
            "size": TOP_K,
            "query": {"knn": {"description_vector": {"vector": vector, "k": TOP_K}}},
            "_source": ["title", "description"],
        },
    )
    hits = res["hits"]["hits"]
    return [
        {"rank": i + 1, "title": h["_source"]["title"], "description": h["_source"]["description"]}
        for i, h in enumerate(hits)
    ]


def _select_rule(question: str, candidates: list[dict]) -> str:
    resp = _get_http().post(
        "/select_rule", json={"question": question, "candidates": candidates}
    )
    resp.raise_for_status()
    return resp.json()["title"]


def _fetch_keywords_from_os(title: str) -> dict:
    res = _get_os().search(
        index=INDEX_NAME,
        body={
            "query": {"term": {"title": title}},
            "_source": ["operator_keywords", "tiebreak_keywords", "negative_keywords"],
        },
    )
    hits = res["hits"]["hits"]
    return hits[0]["_source"] if hits else {}


@observe()
def run(state: AgentState) -> AgentState:
    question = state["question"]

    candidates = _retrieve(question)
    selected = _select_rule(question, candidates)

    if selected == "none":
        langfuse_client().update_current_span(
            input=question,
            output="none",
            metadata={"candidates": candidates},
        )
        return {
            "rule_candidates": candidates,
            "selected_rule": "none",
            "operator_keywords": [],
            "tiebreak_keywords": [],
            "negative_keywords": [],
        }

    keywords = _fetch_keywords_from_os(selected)
    langfuse_client().update_current_span(
        input=question,
        output=selected,
        metadata={
            "candidates": candidates,
            "operator_keywords": keywords.get("operator_keywords", []),
            "tiebreak_keywords": keywords.get("tiebreak_keywords", []),
        },
    )
    return {
        "rule_candidates": candidates,
        "selected_rule": selected,
        "operator_keywords": keywords.get("operator_keywords", []),
        "tiebreak_keywords": keywords.get("tiebreak_keywords", []),
        "negative_keywords": keywords.get("negative_keywords", []),
    }
