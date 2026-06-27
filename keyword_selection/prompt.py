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
    user_prompt = build_user_prompt(query, candidates)
    if answer is not None:
        return user_prompt, answer
    return user_prompt


def format_for_training(query: str, candidates: list[dict], answer: str) -> dict:
    user_prompt, answer = build_prompt(query, candidates, answer)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer},
        ]
    }


# ── 질문 생성용 ───────────────────────────────────────────

QUESTION_GEN_SYSTEM = """You are an expert at generating Korean cafe recommendation questions.
Given a cafe rule title, generate diverse Korean questions that a user would ask to find that type of cafe.

Requirements:
- Generate exactly the requested number of questions
- Half should be colloquial/casual style, half should be formal/polite style
- Colloquial style: short, casual, use abbreviations like "아아"(iced americano), "카공"(cafe study), emoticons like "ㅠㅠ", "~", incomplete sentences
- Formal style: complete sentences, polite endings like "~해주세요", "~있을까요?", "~추천해 주세요"
- Questions must ONLY be about the given title topic, do NOT mix with other cafe attributes
- Output questions only with numbers (e.g., 1. question)"""


def build_question_gen_prompt(title: str, count: int) -> str:
    return f"""Generate {count} diverse Korean cafe recommendation questions for this rule title: "{title}"

- {count // 2} colloquial/casual style questions
- {count // 2} formal/polite style questions
- Questions must strictly relate to "{title}" only"""



# ── Test 데이터 생성용 ────────────────────────────────────

TEST_GEN_SYSTEM = """You are an expert at generating Korean cafe recommendation questions for model evaluation.
Generate diverse questions covering both colloquial and formal styles to ensure comprehensive testing.

Style requirements:
- Half colloquial/casual: short messages, abbreviations like "아아"/"카공", emoticons "ㅠㅠ"/"~", incomplete sentences, context phrases like "오늘 기분전환하고 싶은데"
- Half formal/polite: complete sentences, polite endings like "~해주세요", "~있을까요?", "~추천해 주세요"
- Vary question length and phrasing — avoid repetitive patterns
- Each question must clearly and specifically relate to the given rule title
- Output questions only with numbers (e.g., 1. question)"""


def build_test_gen_prompt(title: str, description: str, count: int) -> str:
    half = count // 2
    return f"""Rule title: "{title}"
Rule description: {description}

Generate {count} diverse Korean cafe recommendation questions for this rule.
- {half} colloquial/casual style
- {half} formal/polite style
Each question must be specifically about "{title}" with varied phrasing."""