from typing import Literal, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    question_type: Literal["non-menu", "menu-only", "menu-complex", "invalid", "fallback"]
    rule_candidates: list[dict]
    selected_rule: str | None
    operator_keywords: list[str]
    tiebreak_keywords: list[str]
    negative_keywords: list[str]
    soar_result: dict | None
    impasse_iterations: int
    impasse_exhausted: bool
    extracted_menu: str | None
    menu_cafe_names: list[str] | None
    rule_fallback_used: bool
    fallback_rules: list[str]
    needs_rule_creation: bool
    rule_created: bool
