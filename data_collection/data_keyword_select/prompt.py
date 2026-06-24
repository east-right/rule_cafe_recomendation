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