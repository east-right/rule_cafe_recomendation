import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe
from langfuse.openai import OpenAI

from service.prompt import MENU_EXTRACT_SYSTEM, MENU_EXTRACT_USER
from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "cafe.db"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _fetch_menu_list() -> list[str]:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT name FROM menus ORDER BY name").fetchall()
    conn.close()
    return [r[0] for r in rows]


def _fetch_cafes_by_menu(menu_name: str) -> list[str]:
    """menu-complex용: 해당 메뉴를 가진 매장 전부 (soar 후보로)."""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT DISTINCT cafe_name FROM cafe_menus WHERE menu_name = ?",
        (menu_name,),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _fetch_top_menu_cafes(menu_name: str, limit: int = 2) -> list[str]:
    """menu-only용: 해당 메뉴 맛집score 상위 N개 매장(긍정만).
    정확 매칭이 없으면 LIKE '%메뉴%'로 fallback (예: 라떼 → 바닐라라떼)."""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        """SELECT cafe_name FROM cafe_menus
           WHERE menu_name = ? AND sentiment = '긍정'
           ORDER BY menu_score DESC LIMIT ?""",
        (menu_name, limit),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            """SELECT cafe_name FROM cafe_menus
               WHERE menu_name LIKE ? AND sentiment = '긍정'
               ORDER BY menu_score DESC LIMIT ?""",
            (f"%{menu_name}%", limit),
        ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _extract_menu(question: str, menu_list: list[str]) -> str | None:
    menu_str = "\n".join(menu_list)
    response = _get_client().chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": MENU_EXTRACT_SYSTEM},
            {"role": "user", "content": MENU_EXTRACT_USER.format(question=question, menu_list=menu_str)},
        ],
        max_tokens=32,
        temperature=0,
    )
    result = response.choices[0].message.content.strip()
    return None if result == "none" else result


@observe()
def run(state: AgentState) -> AgentState:
    question = state["question"]
    question_type = state.get("question_type", "")

    if question_type not in ("menu-only", "menu-complex"):
        return {"extracted_menu": None, "menu_cafe_names": None}

    menu_list = _fetch_menu_list()
    extracted = _extract_menu(question, menu_list)

    if not extracted:
        get_client().update_current_span(
            input={"question": question},
            output={"extracted_menu": None, "fallback": True},
        )
        return {"extracted_menu": None, "menu_cafe_names": None}

    # menu-only: 맛집score 상위 2개(LIKE fallback) / menu-complex: 메뉴 가진 매장 전부
    if question_type == "menu-only":
        cafe_names = _fetch_top_menu_cafes(extracted, limit=2)
    else:  # menu-complex
        cafe_names = _fetch_cafes_by_menu(extracted)

    get_client().update_current_span(
        input={"question": question},
        output={"extracted_menu": extracted, "cafe_count": len(cafe_names)},
    )
    return {
        "extracted_menu": extracted,
        "menu_cafe_names": cafe_names if cafe_names else None,
    }
