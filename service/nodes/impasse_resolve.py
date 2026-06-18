import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from service.prompt import IMPASSE_SYSTEM, IMPASSE_USER
from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "cafe.db"

MAX_IMPASSE_ITER = 5

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _load_keywords(cafe_names: list[str]) -> dict[str, list[str]]:
    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ",".join("?" * len(cafe_names))
    rows = conn.execute(
        f"SELECT cafe_name, keyword FROM cafe_keywords "
        f"WHERE sentiment = '긍정' AND cafe_name IN ({placeholders})",
        cafe_names,
    ).fetchall()
    conn.close()
    result: dict[str, list[str]] = {}
    for name, kw in rows:
        result.setdefault(name, []).append(kw)
    return result


def run(state: AgentState) -> AgentState:
    question = state["question"]
    rule_name = state["selected_rule"]
    operator_keywords = state["operator_keywords"]
    tiebreak_keywords = state["tiebreak_keywords"]
    candidates = state["soar_result"]["candidates"]

    used = set(operator_keywords) | set(tiebreak_keywords)
    cafe_kws = _load_keywords(candidates)

    all_kws: set[str] = set()
    for kws in cafe_kws.values():
        all_kws.update(kws)
    available = sorted(all_kws - used)

    if not available:
        return {"impasse_exhausted": True}

    resp = _get_client().chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": IMPASSE_SYSTEM},
            {"role": "user", "content": IMPASSE_USER.format(
                question=question,
                rule_name=rule_name,
                used_keywords=", ".join(sorted(used)) if used else "없음",
                available_keywords=", ".join(available),
            )},
        ],
        max_tokens=20,
        temperature=0,
    )

    new_kw = resp.choices[0].message.content.strip()

    return {
        "tiebreak_keywords": tiebreak_keywords + [new_kw],
        "impasse_iterations": state.get("impasse_iterations", 0) + 1,
    }
