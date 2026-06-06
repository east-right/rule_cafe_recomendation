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
