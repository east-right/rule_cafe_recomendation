"""
FastAPI 서빙 진입점.

핵심: 모델(임베딩 + sLLM)을 startup에 1회 로드해 상주시키고,
LangGraph 파이프라인을 HTTP로 노출한다. CLI(`python -m service.main`)가
매 실행 콜드스타트를 겪던 구조를 상주 서버로 대체.

실행:
    uv run uvicorn service.api:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from service import main
from service.nodes import rule_select


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델(임베딩+sLLM)은 별도 모델 서버(inference.server)가 로드한다.
    # app은 OpenSearch 클라이언트와 그래프만 미리 준비.
    rule_select._get_os()  # OpenSearch 클라이언트
    main.get_graph()       # LangGraph 컴파일
    yield


app = FastAPI(title="Cafe Recommendation API", lifespan=lifespan)


class RecommendRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """질문 → 추천 답변 토큰 스트리밍."""
    return StreamingResponse(
        main.stream(req.question),
        media_type="text/plain; charset=utf-8",
    )
