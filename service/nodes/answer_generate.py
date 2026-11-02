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


def _fetch_cafe_sentiment_keywords(cafe_name: str) -> dict[str, set[str]]:
    """추천 매장의 긍정/부정 관측 키워드를 분리해서 로드."""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT keyword, sentiment FROM cafe_keywords WHERE cafe_name = ?",
        (cafe_name,),
    ).fetchall()
    conn.close()
    pos: set[str] = set()
    neg: set[str] = set()
    for kw, sentiment in rows:
        if sentiment == "긍정":
            pos.add(kw)
        elif sentiment == "부정":
            neg.add(kw)
    return {"긍정": pos, "부정": neg}


def _extract_grounds(state: AgentState, cafe_name: str) -> dict:
    """추천 매장 기준으로 답변 근거를 감성·충족 여부로 분리한다 (B2).

    - satisfied_operator : 룰 operator 중 이 매장이 '실제 보유'한 긍정 키워드 (미충족 operator 제외)
    - used_tiebreak      : 룰 tiebreak 중 이 매장이 '실제 보유'한 긍정 키워드
    - unobserved_negative: 룰 negative 중 이 매장 부정 리뷰에서 '확인되지 않은' 것
    기존 로직이 부정 키워드(neg_narrowed)와 미충족 operator를 근거로 혼입하던 문제를 제거한다.
    """
    operator_keywords = state.get("operator_keywords", []) or []
    tiebreak_keywords = state.get("tiebreak_keywords", []) or []
    negative_keywords = state.get("negative_keywords", []) or []

    kws = _fetch_cafe_sentiment_keywords(cafe_name)
    pos, neg = kws["긍정"], kws["부정"]

    return {
        "satisfied_operator": [k for k in operator_keywords if k in pos],
        "used_tiebreak": [k for k in tiebreak_keywords if k in pos],
        "unobserved_negative": [k for k in negative_keywords if k not in neg],
    }


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
    grounds = _extract_grounds(state, cafe_name)

    # 긍정 근거: 만족한 operator + 사용된 tiebreak (중복 제거, 순서 보존)
    seen: set[str] = set()
    positive = [k for k in grounds["satisfied_operator"] + grounds["used_tiebreak"]
                if not (k in seen or seen.add(k))]
    unobserved = grounds["unobserved_negative"]

    user_prompt = ANSWER_USER.format(
        question=state["question"],
        cafe_name=cafe_name,
        address=cafe_info.get("address") or "정보 없음",
        status=cafe_info.get("status") or "정보 없음",
        review_summary=cafe_info.get("review_summary") or "정보 없음",
        positive_grounds=", ".join(positive) if positive else "정보 없음",
        unobserved_negative=", ".join(unobserved) if unobserved else "없음",
    )

    get_client().update_current_span(
        input={"question": state["question"], "cafe_name": cafe_name, "grounds": grounds},
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
