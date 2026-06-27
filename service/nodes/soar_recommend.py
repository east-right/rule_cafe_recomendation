import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe

from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "cafe.db"


# ── SQLite 로드 ────────────────────────────────────────────────

def _load_cafe_items(menu_cafe_names: list[str] | None = None) -> dict[str, list[str]]:
    """매장별 키워드 로드 (긍정 operator/tiebreak 매칭 + 부정 narrowing 모두 포함)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    if menu_cafe_names:
        placeholders = ",".join("?" * len(menu_cafe_names))
        rows = conn.execute(
            f"SELECT cafe_name, keyword FROM cafe_keywords WHERE cafe_name IN ({placeholders})",
            menu_cafe_names,
        ).fetchall()
    else:
        rows = conn.execute("SELECT cafe_name, keyword FROM cafe_keywords").fetchall()

    conn.close()

    result: dict[str, list[str]] = {}
    for row in rows:
        result.setdefault(row["cafe_name"], []).append(row["keyword"])
    return result


# ── 규칙 기반 추천 추론 (XAI) ──────────────────────────────────
# Soar 인지아키텍처에서 영감받은 투명한 단계별 추론.
#   operator(후보) → negative(부정 narrowing) → tiebreak(긍정 세부로 우열)
# 각 단계의 remaining을 그대로 기록해 추천 근거(trace)로 노출한다.

def _trace_tiebreak(
    operator_keywords: list[str],
    tiebreak_keywords: list[str],
    negative_keywords: list[str],
    cafe_items: dict[str, list[str]],
    menu_mode: bool = False,
) -> list[dict]:
    if menu_mode:
        # 메뉴 매장 전체가 후보 (operator는 menu 매칭으로 이미 좁혀짐)
        candidates = set(cafe_items.keys())
    else:
        op_set = set(operator_keywords)
        candidates = {c for c, kws in cafe_items.items() if op_set & set(kws)}

    traces = [{"step": "operator", "keyword": None, "remaining": sorted(candidates)}]
    depth = 0

    # negative 먼저: 부정 없는 매장 우선
    for neg_kw in negative_keywords:
        depth += 1
        clean = {c for c in candidates if neg_kw not in set(cafe_items.get(c, []))}
        if 0 < len(clean) < len(candidates):
            candidates = clean
            traces.append({"step": f"S{depth}_neg_narrowed", "keyword": neg_kw, "remaining": sorted(candidates)})
        else:
            traces.append({"step": f"S{depth}_neg_no_effect", "keyword": neg_kw, "remaining": sorted(candidates)})
        if len(candidates) == 1:
            return traces

    # positive: 긍정 세부로 우열
    for tb_kw in tiebreak_keywords:
        depth += 1
        filtered = {c for c in candidates if tb_kw in set(cafe_items.get(c, []))}
        if len(filtered) == 1:
            candidates = filtered
            traces.append({"step": f"S{depth}_resolved", "keyword": tb_kw, "remaining": sorted(candidates)})
            break
        elif len(filtered) > 1:
            candidates = filtered
            traces.append({"step": f"S{depth}_narrowed", "keyword": tb_kw, "remaining": sorted(candidates)})
        else:
            traces.append({"step": f"S{depth}_no_effect", "keyword": tb_kw, "remaining": sorted(candidates)})

    return traces


# ── 노드 진입점 ────────────────────────────────────────────────

@observe()
def run(state: AgentState) -> AgentState:
    title = state.get("selected_rule")
    operator_keywords = state["operator_keywords"]
    tiebreak_keywords = state["tiebreak_keywords"]
    negative_keywords = state.get("negative_keywords", []) or []
    menu_mode = bool(state.get("menu_cafe_names"))

    cafe_items = _load_cafe_items(state.get("menu_cafe_names"))
    trace = _trace_tiebreak(
        operator_keywords, tiebreak_keywords, negative_keywords, cafe_items, menu_mode=menu_mode
    )

    # 마지막 단계의 remaining으로 결정:
    #   1곳 → 추천 / 2곳 이상 못 가름 → impasse(LLM 해소) / 0곳 → no_output
    remaining = trace[-1]["remaining"] if trace else []
    result: dict = {"type": "no_output", "cafe": None, "candidates": [], "trace": trace}
    if len(remaining) == 1:
        result["type"] = "recommendation"
        result["cafe"] = remaining[0]
    elif len(remaining) >= 2:
        result["type"] = "impasse"
        result["candidates"] = remaining[:2]

    get_client().update_current_span(
        input={
            "title": title,
            "operator_keywords": operator_keywords,
            "tiebreak_keywords": tiebreak_keywords,
            "negative_keywords": negative_keywords,
        },
        output=result,
    )
    return {"soar_result": result}
