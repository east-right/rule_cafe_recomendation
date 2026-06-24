import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from langfuse import get_client as langfuse_client, observe

from keyword_selection.search import INDEX_NAME, get_client, load_model, search
from service.prompt import RULE_SELECT_SYSTEM, build_rule_select_user
from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_READ")
SLLM_REPO = "east-right/cafe-keyword-selection-qwen-1.5b"
GGUF_FILENAME = "qwen1.5b-cafe-q8_0.gguf"
LOCAL_GGUF = ROOT / "keyword_selection" / GGUF_FILENAME
TOP_K = 10

_embedding_model = None
_os_client = None
_sllm = None


def _get_retriever():
    global _embedding_model, _os_client
    if _embedding_model is None:
        _embedding_model = load_model()
    if _os_client is None:
        _os_client = get_client()
    return _embedding_model, _os_client


def _gguf_path() -> str:
    """로컬 빌드 산출물이 있으면 사용, 없으면 HF에서 다운로드."""
    if LOCAL_GGUF.exists():
        return str(LOCAL_GGUF)
    return hf_hub_download(SLLM_REPO, GGUF_FILENAME, token=HF_TOKEN)


def _get_sllm() -> Llama:
    global _sllm
    if _sllm is None:
        _sllm = Llama(
            model_path=_gguf_path(),
            n_ctx=4096,
            n_threads=os.cpu_count(),
            verbose=False,
        )
    return _sllm


def _fetch_keywords_from_os(title: str) -> dict:
    _, client = _get_retriever()
    res = client.search(
        index=INDEX_NAME,
        body={
            "query": {"term": {"title": title}},
            "_source": ["operator_keywords", "tiebreak_keywords", "negative_keywords"],
        },
    )
    hits = res["hits"]["hits"]
    return hits[0]["_source"] if hits else {}


def _retrieve(question: str) -> list[dict]:
    model, client = _get_retriever()
    results = search(question, client, model, TOP_K)
    return [
        {"rank": i + 1, "title": r["title"], "description": r["description"]}
        for i, r in enumerate(results)
    ]


def _select_rule(question: str, candidates: list[dict]) -> str:
    llm = _get_sllm()
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": RULE_SELECT_SYSTEM},
            {"role": "user", "content": build_rule_select_user(question, candidates)},
        ],
        max_tokens=32,
        temperature=0.0,
        repeat_penalty=1.1,
    )
    return resp["choices"][0]["message"]["content"].strip()


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
