QUESTION_GENERATION_SYSTEM_PROMPT = """You are a test data generator for a Korean cafe recommendation system.
Given cafe keywords, generate natural Korean questions a user would ask when looking for a cafe.
 
Rules:
1. Questions must request cafe recommendations.
2. Use varied speech styles (formal, informal, casual).
3. Do NOT copy keywords verbatim — rephrase naturally.
4. Each question must use different wording.
5. Output ONLY a JSON array. No other text.
 
Output format:
["question1", "question2", "question3"]"""

RULE_GENERATION_SYSTEM_PROMPT = """You are an expert that generates cafe recommendation rule metadata.

Given a Korean cafe recommendation question, seed keywords, and available keywords by type
(each type split into 긍정 = desirable features, 부정 = features to avoid),
generate a JSON object with exactly 4 fields.

Rules:
1. title: Short Korean label for this rule (2-6 chars, e.g. "공부", "데이트", "반려동물")
2. description: 1-2 Korean sentences describing what kind of cafe this rule targets. Used for semantic search.
3. keywords: List of POSITIVE keywords selected ONLY from the provided 긍정 keywords.
   - Must include all seed keywords
   - Add other relevant keywords from the 긍정 list
   - 3 to 8 keywords total
   - Do NOT invent keywords not in the list
4. negative_keywords: List of keywords this rule should AVOID, selected ONLY from the provided 부정 keywords.
   - Choose ones that clearly conflict with this rule's intent (e.g. "공부" rule avoids "시끄럽다")
   - 0 to 5 keywords (use [] if none are clearly relevant)
   - Do NOT invent keywords not in the 부정 list

Output ONLY valid JSON. No other text.

Example output:
{
  "title": "공부",
  "description": "조용하고 집중하기 좋은 환경을 원하는 카공족을 위한 카페를 추천합니다.",
  "keywords": ["조용한 카페", "콘센트_잘 구비되어있다", "카공", "한적한 분위기"],
  "negative_keywords": ["시끄럽다", "사람 많다"]
}"""

MULTI_RULE_DESCRIPTION_PROMPT = """You are an expert at writing Korean cafe recommendation rule descriptions.

Given a question and combined keywords from multiple rule categories, write a short 1-2 sentence Korean description that captures the combined intent.

Rules:
1. Description should reflect ALL the combined conditions
2. Write in natural Korean
3. Output ONLY the description text, nothing else"""


# ── 독립 조합 multi rule 생성 ──────────────────────────────
# single 속성 두 개를 AND로 묶는 복합 rule을 생성한다.
# 핵심 원칙: 두 속성이 "서로 독립적"이어야 한다.
#  - 독립적 = 한쪽만 추천해도 다른 쪽이 자동으로 만족되지 않음 → 둘 다 명시해야 찾을 수 있음 → multi의 가치
#  - 동의어/근접 = 한쪽 추천으로 다른 쪽도 커버됨 → single로 충분 → multi 불필요
MULTI_RULE_GENERATION_SYSTEM_PROMPT = """You are an expert designing COMPOSITE (AND) rules for a Korean cafe recommendation system.

You are given a list of SINGLE cafe attributes. Combine exactly TWO of them into a composite rule that a real user would request together in one sentence.

A composite rule is only worth creating when the two attributes are INDEPENDENT — recommending a cafe for one does NOT automatically satisfy the other, so BOTH must be stated to find the right cafe.

STRICT RULES:
1. INDEPENDENT only: pick two attributes that do not imply each other.
   - GOOD (independent → must satisfy both): 주차장+공부, 가족+루프탑, 반려동물+디저트, 24시간+빨래방카페
   - BAD (near-synonym → one already covers the other, DO NOT combine): 티타임+차맛집, 독서+독서실, 여유+조용함, 개방감+채광, 아침+아침차
2. NO contradictions: the two must be simultaneously possible in one cafe.
   - BAD: 흡연실+아이동반, 조용함+활기, 독서실+모임
3. REALISTIC: a single person would plausibly ask for both at once.
4. Use a wide variety of attributes — do not reuse the same attribute too often.

For each composite rule output a JSON object:
{"member1": "<attr A>", "member2": "<attr B>", "title": "<short Korean label combining both, 3-7 chars>", "description": "<1 Korean sentence stating BOTH conditions with AND, ending '~ 카페를 찾는 분들을 위한 추천입니다.'>"}

Output ONLY a JSON array of these objects. No other text."""


def build_multi_generation_prompt(single_lines: str, n: int, banned_pairs: str, avoid_titles: str = "") -> str:
    avoid = f"\n\nAlready created (do NOT duplicate these combinations):\n{avoid_titles}" if avoid_titles else ""
    return f"""Single attributes (label - description):
{single_lines}

Near-synonym pairs detected by embedding (these are too similar — DO NOT combine, and avoid combining other obvious synonyms):
{banned_pairs}

Generate {n} INDEPENDENT composite rules following all rules. Each must combine two attributes from the list above by their exact labels.{avoid}"""