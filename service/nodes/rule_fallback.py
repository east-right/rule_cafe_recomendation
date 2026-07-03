"""
룰 선택 폴백 노드.

1차 sLLM(rule_select)이 후보 중 '정확히' 맞는 룰을 못 찾아 'none'을 반환했을 때 동작.
sLLM은 단일 룰이 정확히 매칭될 때만 고르도록 학습돼, 복합·모호한 질문에서 none을 자주 낸다
(예: "조용하고 콘센트 있는 카공 카페" → 단일 룰 없음 → none).

이 노드는 RAG 후보(top-10) 안에서 LLM이 질문에 '가장 가까운' 룰을 강제로 1~3개 고른다.
  - 1순위 룰 → operator (후보 풀)
  - 2~3순위 룰의 operator 키워드 → tiebreak (AND 조건으로 narrowing)
  - negative 키워드는 합집합
→ none으로 죽지 않고 추천까지 이어지며, 고른 근거(키워드)는 그대로 soar_recommend의 trace에 남는다.

키워드는 LLM이 지어내지 않고 OpenSearch의 룰 정의에서 가져온다(환각 방지).
추론은 빠른 gpt-4.1-mini 사용(선택 태스크).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe
from langfuse.openai import OpenAI

from service.nodes.rule_select import _fetch_keywords_from_os
from service.prompt import RULE_FALLBACK_SYSTEM, RULE_FALLBACK_USER
from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL = "gpt-4.1-mini"
MAX_RULES = 3

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _pick_rules(question: str, candidates: list[dict]) -> list[str] | None:
    """후보 중 질문에 가장 가까운 룰 title을 1~MAX_RULES개 선택.

    반환:
        list[str] — 선택된 룰 (Tier 1)
        None      — 맞는 룰이 전혀 없음(NONE_FIT) → 신규 룰 생성(Tier 2)으로
    """
    cand_text = "\n".join(f"- {c['title']}: {c['description']}" for c in candidates)
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RULE_FALLBACK_SYSTEM},
            {"role": "user", "content": RULE_FALLBACK_USER.format(
                question=question, candidates=cand_text
            )},
        ],
        max_tokens=64,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    if "NONE_FIT" in raw:
        return None

    titles = [t.strip() for t in raw.split(",") if t.strip()]
    valid = {c["title"] for c in candidates}
    picked = [t for t in titles if t in valid][:MAX_RULES]
    # LLM이 형식을 어겨 하나도 못 맞히면(NONE_FIT도 아님) RAG 1순위로 안전 폴백
    if not picked:
        picked = [candidates[0]["title"]]
    return picked


@observe()
def run(state: AgentState) -> AgentState:
    question = state["question"]
    candidates = state.get("rule_candidates") or []

    if not candidates:
        # 후보 자체가 없음 → 신규 룰 생성으로 (RAG가 아무것도 못 찾은 경우)
        get_client().update_current_span(input=question, output="no_candidates")
        return {"rule_fallback_used": True, "needs_rule_creation": True}

    picked = _pick_rules(question, candidates)
    if picked is None:
        # 후보 전부 무관(NONE_FIT) → 티어2(신규 룰 생성)로
        get_client().update_current_span(input=question, output="NONE_FIT")
        return {"rule_fallback_used": True, "needs_rule_creation": True}

    primary, *secondary = picked
    primary_kw = _fetch_keywords_from_os(primary)

    operator_keywords = list(primary_kw.get("operator_keywords", []))
    tiebreak_keywords = list(primary_kw.get("tiebreak_keywords", []))
    negative_keywords = list(primary_kw.get("negative_keywords", []))

    # 2순위 이하 룰의 operator 키워드를 tiebreak(AND narrowing)로 흡수
    for title in secondary:
        kw = _fetch_keywords_from_os(title)
        for op in kw.get("operator_keywords", []):
            if op not in operator_keywords and op not in tiebreak_keywords:
                tiebreak_keywords.append(op)
        for neg in kw.get("negative_keywords", []):
            if neg not in negative_keywords:
                negative_keywords.append(neg)

    get_client().update_current_span(
        input={"question": question, "candidates": [c["title"] for c in candidates]},
        output={"picked": picked, "operator": operator_keywords, "tiebreak": tiebreak_keywords},
    )

    return {
        "selected_rule": primary,
        "operator_keywords": operator_keywords,
        "tiebreak_keywords": tiebreak_keywords,
        "negative_keywords": negative_keywords,
        "rule_fallback_used": True,
        "fallback_rules": picked,
    }
