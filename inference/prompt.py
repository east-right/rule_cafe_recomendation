"""모델 서버 전용 프롬프트 (sLLM rule 선택). service 패키지에 의존하지 않는다."""


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
