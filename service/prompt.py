# ── Rule Select ───────────────────────────────────────────────

RULE_SELECT_SYSTEM = """You are a cafe recommendation rule selector.
Given a user query and a list of candidate rules, return the title of the most appropriate rule.
If no rule is appropriate, return "none".
Return only the title. Do not include any explanation."""


def build_rule_select_user(query: str, candidates: list[dict]) -> str:
    candidate_str = "\n".join(
        f"{c['rank']}. {c['title']} - {c['description']}"
        for c in candidates
    )
    return f"[질문]\n{query}\n\n[후보]\n{candidate_str}"


# ── Impasse Resolve ────────────────────────────────────────────

IMPASSE_SYSTEM = """당신은 카페 추천 시스템의 키워드 선택 전문가입니다.

Soar 인지 아키텍처가 카페 추천 중 동점(impasse)을 해결하지 못할 때,
사용자 질문과 현재 룰에 맞는 추가 구별 키워드를 하나 선택합니다.

[제약조건]
- '선택 가능한 키워드' 목록에서만 선택하세요.
- 키워드는 정확히 하나만 출력하세요.
- 설명 없이 키워드만 출력하세요."""

IMPASSE_USER = """[사용자 질문]
{question}

[적용 중인 룰]
{rule_name}

[이미 사용된 키워드]
{used_keywords}

[선택 가능한 키워드]
{available_keywords}

impasse를 해결할 키워드 하나를 선택하세요."""
