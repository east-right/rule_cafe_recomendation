"""
공통 쿼리 함수 모음
"""

from schema import get_connection


def get_cafes_by_keywords(keywords: list[str]) -> dict[str, list[str]]:
    """
    operator_keywords 중 하나라도 가진 매장과 해당 매장의 전체 긍정 키워드 반환
    Soar input-link 주입용

    Returns:
        {"매장명": ["키워드1", "키워드2", ...], ...}
    """
    placeholders = ",".join("?" * len(keywords))
    query = f"""
        SELECT DISTINCT cafe_name
        FROM cafe_keywords
        WHERE keyword IN ({placeholders})
          AND sentiment = '긍정'
    """
    with get_connection() as conn:
        matched = [row["cafe_name"] for row in conn.execute(query, keywords)]

        if not matched:
            return {}

        ph2 = ",".join("?" * len(matched))
        kw_rows = conn.execute(
            f"""SELECT cafe_name, keyword
                FROM cafe_keywords
                WHERE cafe_name IN ({ph2})
                  AND sentiment = '긍정'""",
            matched,
        ).fetchall()

    result: dict[str, list[str]] = {name: [] for name in matched}
    for row in kw_rows:
        result[row["cafe_name"]].append(row["keyword"])

    return result


def get_cafe_info(cafe_name: str) -> dict | None:
    """
    최종 LLM 답변용 매장 기본 정보 조회
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT name, address, status, review_summary FROM cafes WHERE name = ?",
            (cafe_name,),
        ).fetchone()
    return dict(row) if row else None
