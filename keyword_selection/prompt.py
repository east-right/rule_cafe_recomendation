SYSTEM_PROMPT = """You are a cafe recommendation rule selector.
Given a user query and a list of candidate rules, return the title of the most appropriate rule.
If no rule is appropriate, return "none".
Return only the title. Do not include any explanation."""


def build_user_prompt(query: str, candidates: list[dict]) -> str:
    candidate_str = "\n".join(
        f"{c['rank']}. {c['title']} - {c['description']}"
        for c in candidates
    )
    return f"[질문]\n{query}\n\n[후보]\n{candidate_str}"


def build_prompt(query: str, candidates: list[dict], answer: str | None = None) -> str | tuple[str, str]:
    """
    학습용: answer 있으면 (user_prompt, answer) 반환
    추론용: answer 없으면 user_prompt만 반환
    """
    user_prompt = build_user_prompt(query, candidates)

    if answer is not None:
        return user_prompt, answer
    return user_prompt


def format_for_training(query: str, candidates: list[dict], answer: str) -> dict:
    """
    Unsloth SFTTrainer용 chat 형식 변환
    """
    user_prompt, answer = build_prompt(query, candidates, answer)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer},
        ]
    }


if __name__ == "__main__":
    # 테스트
    sample_candidates = [
        {"rank": 1, "title": "카공족", "description": "공부나 작업하기 좋은 환경을 갖춘 카페를 추천합니다."},
        {"rank": 2, "title": "넓은공간", "description": "조용한 분위기에서 넉넉한 공간을 갖춘 카페를 추천합니다."},
        {"rank": 3, "title": "편안함", "description": "자리 넓고 편안한 분위기에서 휴식이나 대화를 즐기기에 좋은 카페를 추천합니다."},
    ]

    result = format_for_training(
        query="조용하고 콘센트 있는 카페 추천해줘",
        candidates=sample_candidates,
        answer="카공족",
    )

    for msg in result["messages"]:
        print(f"[{msg['role']}]")
        print(msg["content"])
        print()