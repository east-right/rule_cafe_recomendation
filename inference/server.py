"""
모델 추론 서버 (CPU 백엔드).

임베딩(BGE-M3)과 sLLM(Qwen2.5-1.5B gguf)만 로드해 HTTP로 노출한다.
LangGraph app(rule_select)이 이 서버를 호출하고, OpenSearch 검색은 app이 담당.
→ 모델(GPU 교체 대상)과 인프라(OpenSearch)를 분리하는 것이 목적.

GPU 트랙에서는 이 서버의 sLLM 백엔드를 vLLM으로 교체한다(인터페이스 동일).

실행:
    uv run uvicorn inference.server:app --host 0.0.0.0 --port 8001
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from inference.prompt import RULE_SELECT_SYSTEM, build_rule_select_user

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_READ")
EMBED_MODEL = "east-right/bge-m3-cafe-finetuned"
SLLM_REPO = "east-right/cafe-keyword-selection-qwen-1.5b"
GGUF_FILENAME = "qwen1.5b-cafe-q8_0.gguf"
LOCAL_GGUF = ROOT / "keyword_selection" / GGUF_FILENAME

BACKEND = "cpu"

_embedder: SentenceTransformer | None = None
_llm: Llama | None = None


def _gguf_path() -> str:
    """로컬 빌드 산출물 우선, 없으면 HF에서 다운로드."""
    if LOCAL_GGUF.exists():
        return str(LOCAL_GGUF)
    return hf_hub_download(SLLM_REPO, GGUF_FILENAME, token=HF_TOKEN)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _llm
    _embedder = SentenceTransformer(EMBED_MODEL, token=HF_TOKEN)
    _llm = Llama(
        model_path=_gguf_path(),
        n_ctx=4096,
        n_threads=os.cpu_count(),
        verbose=False,
    )
    yield


app = FastAPI(title="Cafe Inference Server", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    vectors: list[list[float]]


class SelectRuleRequest(BaseModel):
    question: str
    candidates: list[dict]


class SelectRuleResponse(BaseModel):
    title: str


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND, "models": [EMBED_MODEL, GGUF_FILENAME]}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """텍스트 → 정규화 임베딩 벡터."""
    vectors = _embedder.encode(req.texts, normalize_embeddings=True)
    return {"vectors": [v.tolist() for v in vectors]}


@app.post("/select_rule", response_model=SelectRuleResponse)
def select_rule(req: SelectRuleRequest):
    """질문 + 후보 rule → 선택된 rule title(또는 'none')."""
    resp = _llm.create_chat_completion(
        messages=[
            {"role": "system", "content": RULE_SELECT_SYSTEM},
            {"role": "user", "content": build_rule_select_user(req.question, req.candidates)},
        ],
        max_tokens=32,
        temperature=0.0,
        repeat_penalty=1.1,
    )
    return {"title": resp["choices"][0]["message"]["content"].strip()}
