import os
import sqlite3
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from langfuse import get_client, observe
from langfuse.openai import OpenAI

from service.prompt import ANSWER_SYSTEM, ANSWER_USER
from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "cafe.db"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _fetch_cafe_info(cafe_name: str) -> dict:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT name, address, status, review_summary FROM cafes WHERE name = ?",
        (cafe_name,),
    ).fetchone()
    conn.close()
    return dict(row) if row else {}


def _extract_effective_keywords(state: AgentState) -> list[str]:
    operator_keywords = state.get("operator_keywords", [])
    trace = (state.get("soar_result") or {}).get("trace", [])

    effective_tiebreak = [
        t["keyword"]
        for t in trace
        if t.get("keyword") and "no_effect" not in t["step"]
    ]
    return operator_keywords + effective_tiebreak


def _stream_menu_only(state: AgentState) -> Iterator[str]:
    """menu-only: 맛집score 상위 매장 2곳을 모두 추천."""
    cafe_names = state.get("menu_cafe_names") or []
    if not cafe_names:
        yield "추천할 카페를 찾지 못했습니다."
        return

    menu = state.get("extracted_menu") or ""
    infos = [_fetch_cafe_info(c) for c in cafe_names]
    cafes_text = "\n".join(
        f"- {i['name']} (주소: {i.get('address') or '정보 없음'}, 영업: {i.get('status') or '정보 없음'})"
        f"\n  리뷰: {i.get('review_summary') or '정보 없음'}"
        for i in infos if i
    )
    user_prompt = (
        f"[사용자 질문]\n{state['question']}\n\n"
        f"[추천 메뉴]\n{menu}\n\n"
        f"[맛집 후보 (맛집score 상위)]\n{cafes_text}\n\n"
        f"위 메뉴의 맛집 {len(infos)}곳을 모두 추천하는 답변을 작성하세요."
    )

    get_client().update_current_span(
        input={"question": state["question"], "menu": menu, "cafes": cafe_names},
    )
    response = _get_client().chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=400,
        temperature=0.7,
        stream=True,
    )
    full_text = ""
    for chunk in response:
        token = chunk.choices[0].delta.content or ""
        if token:
            full_text += token
            yield token
    get_client().update_current_span(output=full_text)


@observe()
def stream(state: AgentState) -> Iterator[str]:
    if state.get("question_type") in ("invalid", "fallback"):
        yield "답변할 수 없는 내용입니다. 카페의 추천에 대한 답변할 수 있도록 질문을 해주세요."
        return

    # menu-only는 맛집 2곳을 모두 추천 (soar 우회)
    if state.get("question_type") == "menu-only":
        yield from _stream_menu_only(state)
        return

    soar_result = state.get("soar_result") or {}
    cafe_name = soar_result.get("cafe")

    if not cafe_name:
        yield "추천할 카페를 찾지 못했습니다."
        return

    cafe_info = _fetch_cafe_info(cafe_name)
    keywords = _extract_effective_keywords(state)

    user_prompt = ANSWER_USER.format(
        question=state["question"],
        cafe_name=cafe_name,
        address=cafe_info.get("address") or "정보 없음",
        status=cafe_info.get("status") or "정보 없음",
        review_summary=cafe_info.get("review_summary") or "정보 없음",
        keywords=", ".join(keywords) if keywords else "정보 없음",
    )

    get_client().update_current_span(
        input={"question": state["question"], "cafe_name": cafe_name, "keywords": keywords},
        metadata={"cafe_info": cafe_info},
    )

    response = _get_client().chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
        temperature=0.7,
        stream=True,
    )

    full_text = ""
    for chunk in response:
        token = chunk.choices[0].delta.content or ""
        if token:
            full_text += token
            yield token

    get_client().update_current_span(output=full_text)
