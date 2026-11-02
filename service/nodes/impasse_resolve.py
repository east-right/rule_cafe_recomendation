import os
import sqlite3
import threading
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe
from langfuse.openai import OpenAI

from keyword_selection.search import INDEX_NAME, get_client as os_get_client
from service.prompt import IMPASSE_SYSTEM, IMPASSE_USER
from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "cafe.db"

MAX_IMPASSE_ITER = 5

# B3: impasse 선택 키워드를 OpenSearch 룰에 영구 저장할지. 기본 off(요청 단위 임시 적용).
PERSIST_IMPASSE_KEYWORD = os.getenv("IMPASSE_PERSIST", "false").lower() == "true"

_client: OpenAI | None = None

_APPEND_SCRIPT = (
    "if (ctx._source.tiebreak_keywords == null) {"
    "  ctx._source.tiebreak_keywords = [params.kw]"
    "} else if (!ctx._source.tiebreak_keywords.contains(params.kw)) {"
    "  ctx._source.tiebreak_keywords.add(params.kw)"
    "}"
)


def _update_os_tiebreak(title: str, keyword: str) -> None:
    try:
        res = os_get_client().update_by_query(
            index=INDEX_NAME,
            body={
                "query": {"term": {"title": title}},
                "script": {
                    "source": _APPEND_SCRIPT,
                    "lang": "painless",
                    "params": {"kw": keyword},
                },
            },
        )
        print(f"[OS update] title={title!r} keyword={keyword!r} updated={res.get('updated')} total={res.get('total')}")
    except Exception as e:
        print(f"[OS update error] {e}")


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


@observe()
def run(state: AgentState) -> AgentState:
    question = state["question"]
    rule_name = state["selected_rule"]
    operator_keywords = state["operator_keywords"]
    tiebreak_keywords = state["tiebreak_keywords"]
    candidates = state["soar_result"]["candidates"]

    used = set(operator_keywords) | set(tiebreak_keywords)
    cafe_kws = _load_keywords(candidates)

    # B3: 구별력 검증 — 동점 후보를 '실제로 가르는' 키워드만 후보로.
    #   후보 전체가 보유(c==n) 하거나 아무도 없으면(c==0) narrowing 불가 → 제외.
    n = len(candidates)
    counts: Counter[str] = Counter()
    for kws in cafe_kws.values():
        counts.update(set(kws))
    discriminating = sorted(k for k, c in counts.items() if k not in used and 0 < c < n)

    if not discriminating:
        # 가를 수 있는 키워드가 없음 → 정직하게 교착 종료(억지 추천 금지)
        get_client().update_current_span(
            input={"question": question, "candidates": candidates},
            output="exhausted:no_discriminating_keyword",
        )
        return {"impasse_exhausted": True}

    resp = _get_client().chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": IMPASSE_SYSTEM},
            {"role": "user", "content": IMPASSE_USER.format(
                question=question,
                rule_name=rule_name,
                used_keywords=", ".join(sorted(used)) if used else "없음",
                available_keywords=", ".join(discriminating),
            )},
        ],
        max_tokens=20,
        temperature=0,
    )

    raw = resp.choices[0].message.content.strip()
    # B3: 허용목록 검증 — LLM 출력이 구별력 키워드 집합 밖이면 결정적 폴백(환각 차단).
    new_kw = raw if raw in discriminating else discriminating[0]

    # B3: 영구 저장은 기본 off(요청 단위 임시 적용). 플래그가 켜졌을 때만 OpenSearch 반영.
    if PERSIST_IMPASSE_KEYWORD:
        threading.Thread(
            target=_update_os_tiebreak, args=(rule_name, new_kw), daemon=False
        ).start()

    iteration = state.get("impasse_iterations", 0) + 1
    get_client().update_current_span(
        input={"question": question, "used_keywords": sorted(used),
               "discriminating_keywords": discriminating},
        output={"raw": raw, "chosen": new_kw, "allowlist_fallback": raw not in discriminating,
                "persisted": PERSIST_IMPASSE_KEYWORD},
        metadata={"iteration": iteration, "candidates": candidates},
    )
    return {
        "tiebreak_keywords": tiebreak_keywords + [new_kw],
        "impasse_iterations": iteration,
    }
