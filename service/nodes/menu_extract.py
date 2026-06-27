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
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT DISTINCT cafe_name FROM cafe_menus WHERE menu_name = ?",
        (menu_name,),
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

    cafe_names = _fetch_cafes_by_menu(extracted)

    get_client().update_current_span(
        input={"question": question},
        output={"extracted_menu": extracted, "cafe_count": len(cafe_names)},
    )
    return {
        "extracted_menu": extracted,
        "menu_cafe_names": cafe_names if cafe_names else None,
    }
