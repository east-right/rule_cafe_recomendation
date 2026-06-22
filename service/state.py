from typing import Literal, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    question_type: Literal["non-menu", "menu-only", "menu-complex", "invalid", "fallback"]
    rule_candidates: list[dict]
    selected_rule: str | None
    operator_keywords: list[str]
    tiebreak_keywords: list[str]
    soar_result: dict | None
