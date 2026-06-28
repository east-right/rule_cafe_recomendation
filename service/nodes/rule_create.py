"""
신규 룰 생성 노드 (티어2, CogRec식 온라인 학습).

rule_fallback이 'NONE_FIT'(후보 룰 전부 질문과 무관)을 판단했을 때 동작.
기존 룰로는 못 푸는 질문이므로, LLM이 질문에 맞는 '새 룰'을 직접 만든다.

핵심 제약:
- 생성 키워드는 반드시 실제 cafe_keywords 어휘 안에서만(환각하면 trace가 0매장 → 빈 추천).
- operator(핵심)는 긍정 어휘, negative는 부정 어휘에서.

생성한 룰은 OpenSearch(cafe_rules)에 영구 저장한다 → 다음에 비슷한 질문이 오면
RAG가 이 룰을 바로 후보로 찾는다(= 온라인 학습). 임베딩은 모델서버(/embed, 동일 BGE-M3).

operator 키워드를 하나도 못 만들면(어휘에 매칭 불가) 룰 생성 실패 →
selected_rule을 none으로 남겨 정직하게 '추천 없음'으로 끝낸다(억지 추천 금지).
"""

import json
import sqlite3
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe

from keyword_selection.search import INDEX_NAME
from service.nodes.rule_fallback import _get_client
from service.nodes.rule_select import _embed, _get_os
from service.prompt import RULE_CREATE_SYSTEM, RULE_CREATE_USER
from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "cafe.db"
MODEL = "gpt-4.1-mini"


def _load_vocab() -> tuple[set[str], set[str]]:
    """cafe_keywords의 실제 키워드 어휘를 긍정/부정으로 로드."""
    conn = sqlite3.connect(str(DB_PATH))
    pos = {r[0] for r in conn.execute(
        "SELECT DISTINCT keyword FROM cafe_keywords WHERE sentiment='긍정'")}
    neg = {r[0] for r in conn.execute(
        "SELECT DISTINCT keyword FROM cafe_keywords WHERE sentiment='부정'")}
    conn.close()
    return pos, neg


def _generate_rule(question: str, pos: set[str], neg: set[str]) -> dict:
    """LLM이 어휘 안에서 신규 룰(JSON) 생성."""
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RULE_CREATE_SYSTEM},
            {"role": "user", "content": RULE_CREATE_USER.format(
                question=question,
                positive_vocab=", ".join(sorted(pos)),
                negative_vocab=", ".join(sorted(neg)),
            )},
        ],
        max_tokens=300,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def _persist_rule(title: str, description: str, operator: list[str],
                  tiebreak: list[str], negative: list[str]) -> None:
    """생성한 룰을 cafe_rules에 인덱싱(임베딩은 모델서버) → 다음 질문부터 RAG가 찾음."""
    vector = _embed(description)
    doc = {
        "title": title,
        "description": description,
        "description_vector": vector,
        "operator_keywords": operator,
        "tiebreak_keywords": tiebreak,
        "negative_keywords": negative,
        "generated": True,  # 자동 생성 룰 표식(추후 검수/정리용)
    }
    _get_os().index(index=INDEX_NAME, id=f"gen-{uuid.uuid4().hex[:12]}", body=doc, refresh=True)


@observe()
def run(state: AgentState) -> AgentState:
    question = state["question"]
    pos, neg = _load_vocab()

    rule = _generate_rule(question, pos, neg)
    title = (rule.get("title") or "").strip()
    description = (rule.get("description") or "").strip()

    # 어휘에 실제로 있는 키워드만 채택(환각 제거)
    operator = [k for k in rule.get("operator_keywords", []) if k in pos]
    tiebreak = [k for k in rule.get("tiebreak_keywords", []) if k in pos]
    negative = [k for k in rule.get("negative_keywords", []) if k in neg]

    # 핵심(operator)을 못 만들면 룰 성립 불가 → 정직하게 추천 없음
    if not operator or not title:
        get_client().update_current_span(
            input=question, output="creation_failed", metadata={"raw": rule}
        )
        return {"rule_created": False}

    _persist_rule(title, description, operator, tiebreak, negative)

    get_client().update_current_span(
        input={"question": question},
        output={"title": title, "operator": operator, "tiebreak": tiebreak, "negative": negative},
        metadata={"description": description, "persisted": True},
    )
    return {
        "selected_rule": title,
        "operator_keywords": operator,
        "tiebreak_keywords": tiebreak,
        "negative_keywords": negative,
        "rule_created": True,
        "fallback_rules": [title],
    }
