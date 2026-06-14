# soar/LLM_keyword/prompt.py

RULE_KEYWORD_EXTRACTION_SYSTEM = """당신은 카페 추천 시스템의 SOAR 규칙을 설계하는 전문가입니다.

주어진 Rule의 title, description, keywords를 분석하여 아래 두 가지를 선택하세요.

1. operator_keywords (4~5개): 이 Rule의 핵심을 가장 잘 나타내는 키워드
   - 이 키워드 중 하나라도 가진 카페가 추천 후보가 됩니다 (OR 조건)
   - Rule의 본질적인 의미를 가장 직접적으로 표현하는 키워드여야 합니다

2. tiebreak_keywords (4~5개): 후보 카페들 사이의 우열을 가릴 수 있는 키워드
   - 리스트 순서가 곧 우선순위입니다 (index 0 = S1, index 1 = S2, ...)
   - operator_keywords보다 더 구체적이거나 강한 신호를 가진 키워드여야 합니다
   - 반드시 제공된 keywords 목록 안에서만 선택하세요

반드시 아래 JSON 형식으로만 응답하세요. 다른 설명은 절대 포함하지 마세요.
{
    "title": "Rule 제목",
    "operator_keywords": ["키워드1", "키워드2"],
    "tiebreak_keywords": ["키워드1", "키워드2"]
}"""


RULE_KEYWORD_EXTRACTION_USER = """Rule 정보:
title: {title}
description: {description}
keywords: {keywords}"""