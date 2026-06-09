MENU_COMPLEX_SYSTEM = """당신은 카페 추천 질문 생성 전문가입니다.
주어진 메뉴 키워드와 비메뉴 키워드를 자연스럽게 조합하여 카페 추천 질문을 생성하세요.

규칙:
- 메뉴 조건과 비메뉴 조건이 반드시 모두 포함되어야 합니다.
- 자연스러운 한국어 구어체로 작성하세요.
- 질문 형식으로 끝나야 합니다. (예: ~해줘, ~알려줘, ~있을까?, ~추천해줘)
- 질문만 출력하고 다른 설명은 하지 마세요."""


def build_menu_complex_prompt(menu_keyword: str, non_menu_keywords: list[str]) -> str:
    non_menu_str = ", ".join(non_menu_keywords)
    return f"메뉴 키워드: {menu_keyword}\n비메뉴 키워드: {non_menu_str}\n\n위 조건을 모두 포함한 자연스러운 카페 추천 질문을 1개 생성해주세요."


# ── Invalid 질문 생성용 ────────────────────────────────────

INVALID_SYSTEM = """당신은 카페 추천 시스템의 테스트 데이터 생성 전문가입니다.
카페 추천 시스템이 처리할 수 없는 질문들을 생성하세요.

카페 추천 시스템이 처리할 수 없는 질문의 종류:
1. 카페가 아닌 다른 업종 추천 요청 (식당, 술집, 편의점, 노래방, 마트 등)
2. 카페 관련이지만 추천이 아닌 질문 (영업시간, 와이파이 비번, 주차 요금, 예약 방법 등)
3. 카페와 완전히 무관한 일상 질문 (날씨, 영화, 주식, 교통, 요리 등)

규칙:
- 자연스러운 한국어 구어체로 작성하세요.
- 실제 사용자가 챗봇에 입력할 법한 질문이어야 합니다.
- 질문만 출력하고 다른 설명은 하지 마세요.
- 1번과 2번 유형에 특히 집중하세요."""


INVALID_USER_TEMPLATE = """다음 유형별로 카페 추천 시스템이 처리할 수 없는 질문을 각각 생성해주세요.

유형 1 (카페 외 업종 추천): {type1_count}개
유형 2 (카페 관련이지만 추천 불가): {type2_count}개
유형 3 (카페와 완전 무관): {type3_count}개

반드시 번호와 유형을 포함하여 출력하세요.
예시:
[유형1] 근처 분위기 좋은 식당 추천해줘
[유형2] 이 카페 몇시에 문 닫아?
[유형3] 오늘 날씨 어때?"""


def build_invalid_prompt(type1_count: int, type2_count: int, type3_count: int) -> str:
    return INVALID_USER_TEMPLATE.format(
        type1_count=type1_count,
        type2_count=type2_count,
        type3_count=type3_count,
    )