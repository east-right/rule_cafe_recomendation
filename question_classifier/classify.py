"""
질문 유형 분류 (OpenAI 기반).

기존 koelectra 파인튜닝 모델을 대체. 사용자 질문을 4가지로 분류한다.
- non-menu     : 메뉴 언급 없이 분위기/시설/대상/위치 조건만
- menu-only    : 특정 메뉴(음료/디저트)만, 비메뉴 조건 없음
- menu-complex : 메뉴 조건 + 비메뉴 조건 모두 포함
- invalid      : 카페 추천과 무관하거나 추천이 아닌 질문

service/nodes/question_valid.py가 classify_question()을 호출해 사용한다.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4.1-mini"
LABELS = {"non-menu", "menu-only", "menu-complex", "invalid"}

QUESTION_CLASSIFY_SYSTEM = """당신은 카페 추천 챗봇의 질문 분류기입니다.
사용자 질문을 아래 4가지 라벨 중 정확히 하나로 분류하세요.

[라벨 정의]
- non-menu: 메뉴 언급 없이 분위기·시설·대상·위치 등 조건만 요청
  예) "조용한 카페 추천", "주차되는 카페", "카공하기 좋은 곳", "혼자 가기 좋은 데", "넓고 아늑한 카페"
- menu-only: 특정 메뉴(음료/디저트)만 요청하고 비메뉴 조건은 없음
  예) "마카롱 맛집", "라떼 맛있는 곳", "아아 잘하는 카페", "크로플 유명한 데"
- menu-complex: 메뉴 조건과 비메뉴 조건을 모두 포함
  예) "마카롱 맛집인데 조용한 카페", "라떼 맛있고 주차되는 곳", "케이크 맛집이면서 공부하기 좋은 데"
- invalid: 카페 추천과 무관하거나, 카페 관련이어도 추천 요청이 아닌 질문
  예) "근처 식당 추천", "이 카페 몇 시에 닫아?", "와이파이 비번 뭐야?", "오늘 날씨 어때?"

[판단 기준]
- 메뉴 = 카페에서 파는 음료/디저트 (아메리카노, 라떼, 마카롱, 케이크, 크로플, 빙수 등)
- 분위기/시설/대상(조용함, 주차, 콘센트, 카공, 데이트, 넓은 좌석 등)은 비메뉴 조건
- 메뉴와 비메뉴가 둘 다 있으면 menu-complex, 메뉴만 있으면 menu-only, 메뉴가 없으면 non-menu
- 카페 추천 의도가 아니면 invalid

[출력]
위 4개 라벨(non-menu / menu-only / menu-complex / invalid) 중 하나만 출력하세요. 다른 말은 절대 하지 마세요."""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def classify_question(question: str) -> str:
    """질문을 4개 라벨 중 하나로 분류. 알 수 없는 응답은 invalid로 처리."""
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": QUESTION_CLASSIFY_SYSTEM},
            {"role": "user", "content": question},
        ],
        max_tokens=8,
        temperature=0,
    )
    label = response.choices[0].message.content.strip().lower()
    return label if label in LABELS else "invalid"
