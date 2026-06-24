import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe

from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SOAR_HOME = os.getenv("SOAR_HOME") or str(ROOT / "soar" / "sml" / "Soar" / "out")
DB_PATH = ROOT / "data" / "cafe.db"

os.add_dll_directory(SOAR_HOME)
sys.path.insert(0, SOAR_HOME)
import Python_sml_ClientInterface as sml  # noqa: E402

# ── Soar 룰 생성 ──────────────────────────────────────────────

_BOILERPLATE = """sp {elaborate*top-state*top-state
   (state <s> ^superstate nil)
-->
   (<s> ^top-state <s>)
}

sp {elaborate*state*top-state
   (state <s> ^superstate.top-state <ts>)
-->
   (<s> ^top-state <ts>)
}

sp {elaborate*state*item*down
   (state <s> ^superstate.item <i>)
-->
   (<s> ^item <i>)
}

sp {apply*recommend-cafe*send-to-python
   (state <s> ^operator <o>
              ^io.output-link <ol>)
   (<o> ^name recommend-cafe
        ^cafe-name <c-name>)
-->
   (<ol> ^final-recommendation <c-name>)
}
"""


def _build_kw_map(
    operator_keywords: list[str],
    tiebreak_keywords: list[str],
    negative_keywords: list[str] | None = None,
) -> dict[str, str]:
    """한국어 키워드 → ASCII ID 매핑 (Soar 파서 인코딩 이슈 회피).
    negative 키워드도 포함해 부정 judge가 ASCII id로 매칭되게 한다."""
    negative_keywords = negative_keywords or []
    all_kw = list(dict.fromkeys(operator_keywords + tiebreak_keywords + negative_keywords))
    return {kw: f"kw{i}" for i, kw in enumerate(all_kw)}


def _make_operator(op_ids: list[str]) -> str:
    kw_list = " ".join(op_ids)
    return f"""sp {{recommend-OPERATOR
   (state <s> ^io.input-link <il>)
   (<il> ^cafe <c>)
   (<c> ^name <c-name> ^keyword << {kw_list} >>)
-->
   (<s> ^operator <o> +)
   (<o> ^name recommend-cafe ^cafe-name <c-name>)
}}
"""


def _make_menu_operator() -> str:
    return """sp {recommend-OPERATOR
   (state <s> ^io.input-link <il>)
   (<il> ^cafe <c>)
   (<c> ^name <c-name> ^has-menu true)
-->
   (<s> ^operator <o> +)
   (<o> ^name recommend-cafe ^cafe-name <c-name>)
}
"""


def _make_s1_judge(kw_id: str) -> str:
    return f"""sp {{recommend-S1-judge
   (state <s> ^operator <o1> +
              ^operator <o2> +
              ^io.input-link <il>)
   (<o1> ^name recommend-cafe ^cafe-name <c1-name>)
   (<o2> ^name recommend-cafe ^cafe-name <c2-name> <> <c1-name>)
   (<il> ^cafe <c1> ^cafe <c2>)
   (<c1> ^name <c1-name> ^keyword {kw_id})
   (<c2> ^name <c2-name>)
   - {{ (<c2> ^keyword {kw_id}) }}
-->
   (<s> ^operator <o1> > <o2>)
}}
"""


def _make_sn_judge(kw_id: str, depth: int) -> str:
    # fires in Soar's S{depth} (depth-1 levels deep from top state S1)
    chain = "".join(
        f"   (<s{i}> ^superstate nil)\n" if i == 1
        else f"   (<s{i}> ^superstate <s{i-1}>)\n"
        for i in range(1, depth)  # s1..s{depth-1}
    )
    return f"""sp {{resolve-tie-S{depth}-judge
   (state <s> ^impasse <any-impasse>
              ^superstate <s{depth-1}>
              ^item <o1> ^item <o2>
              ^top-state <ts>)
{chain}   (<o1> ^name recommend-cafe ^cafe-name <c1-name>)
   (<o2> ^name recommend-cafe ^cafe-name <c2-name> <> <c1-name>)
   (<ts> ^io.input-link <il>)
   (<il> ^cafe <c1> ^cafe <c2>)
   (<c1> ^name <c1-name> ^keyword {kw_id})
   (<c2> ^name <c2-name>)
   - {{ (<c2> ^keyword {kw_id}) }}
-->
   (<s1> ^operator <o1> > <o2>)
}}
"""


def _filter_negative(
    cafe_items: dict[str, list[str]],
    operator_keywords: list[str],
    negative_keywords: list[str],
) -> dict[str, list[str]]:
    """부정 매장을 후보에서 제거(후순위 효과). Soar 룰(worst/페어)이 추론을 깨뜨려
    no_output/무한 impasse를 내므로, Soar 전에 파이썬으로 처리한다.
    부정 없는 operator 후보가 하나라도 있으면 부정 보유 후보를 제거하고,
    전부 부정이면 그대로 둔다(탈락이 아닌 후순위)."""
    if not negative_keywords:
        return cafe_items
    op_set, neg_set = set(operator_keywords), set(negative_keywords)
    op_cands = {c for c, kws in cafe_items.items() if op_set & set(kws)}
    clean = {c for c in op_cands if not (neg_set & set(cafe_items[c]))}
    if not clean:
        return cafe_items  # 전부 부정 → 후순위 불가, 그대로
    dirty = op_cands - clean
    return {c: kws for c, kws in cafe_items.items() if c not in dirty}


def _make_llm_fallback(depth: int) -> str:
    # fires in Soar's S{depth} (depth-1 levels deep from top state S1)
    chain = "".join(
        f"   (<s{i}> ^superstate nil)\n" if i == 1
        else f"   (<s{i}> ^superstate <s{i-1}>)\n"
        for i in range(1, depth)  # s1..s{depth-1}
    )
    return f"""sp {{resolve-tie-S{depth}-ask-llm
   (state <s> ^impasse <any-impasse>
              ^superstate <s{depth-1}>
              ^item <o1> ^item <o2>
              ^top-state <ts>)
{chain}   (<o1> ^cafe-name <c1-name>)
   (<o2> ^cafe-name {{ <c2-name> <> <c1-name> }})
   (<ts> ^io.output-link <ol>)
-->
   (<ol> ^ask-llm <req>)
   (<req> ^cand1 <c1-name>
          ^cand2 <c2-name>
          ^stopped-depth {depth})
}}
"""


def _build_soar_rule(
    operator_keywords: list[str],
    tiebreak_keywords: list[str],
    kw_map: dict[str, str],
    negative_keywords: list[str] | None = None,
    menu_mode: bool = False,
) -> str:
    negative_keywords = negative_keywords or []
    parts = [
        _BOILERPLATE,
        _make_menu_operator() if menu_mode
        else _make_operator([kw_map[kw] for kw in operator_keywords]),
    ]

    # negative는 Soar 룰로 처리하지 않는다. worst/페어 모두 Soar 추론을 깨뜨려
    # (no_output / 무한 impasse), 대신 run()에서 부정 매장을 후보 단계 필터로 거른다.

    # positive judge(impasse 단계) → LLM fallback
    depth = 0
    for kw in tiebreak_keywords:
        depth += 1
        kw_id = kw_map[kw]
        parts.append(_make_s1_judge(kw_id) if depth == 1 else _make_sn_judge(kw_id, depth))

    parts.append(_make_llm_fallback(depth + 1))
    return "\n".join(parts)


# ── SQLite 로드 ────────────────────────────────────────────────

def _load_cafe_items(menu_cafe_names: list[str] | None = None) -> dict[str, list[str]]:
    """매장별 키워드 로드. 긍정(operator/tiebreak 매칭)과 부정(negative judge 매칭)을
    모두 가져온다 — kw_map에 있는 키워드만 Soar로 전달되므로 섞여도 무방하다."""
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


# ── tiebreak 단계별 추적 ──────────────────────────────────────

def _trace_tiebreak(
    operator_keywords: list[str],
    tiebreak_keywords: list[str],
    negative_keywords: list[str],
    cafe_items: dict[str, list[str]],
) -> list[dict]:
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


# ── output-link 파싱 ───────────────────────────────────────────

def _parse_output(agent) -> dict:
    result = {"type": "no_output", "cafe": None, "candidates": []}

    output_link = agent.GetOutputLink()
    if output_link is None:
        return result

    cafe_name = output_link.GetParameterValue("final-recommendation")
    if cafe_name:
        result["type"] = "recommendation"
        result["cafe"] = cafe_name
        return result

    num_commands = agent.GetNumberCommands()
    for i in range(num_commands):
        command = agent.GetCommand(i)
        if command.GetCommandName() == "ask-llm":
            result["type"] = "impasse"
            cand1 = command.GetParameterValue("cand1")
            cand2 = command.GetParameterValue("cand2")
            if cand1:
                result["candidates"].append(cand1)
            if cand2:
                result["candidates"].append(cand2)

    if result["type"] == "impasse":
        result["candidates"] = list(dict.fromkeys(result["candidates"]))
    return result


# ── 노드 진입점 ────────────────────────────────────────────────

@observe()
def run(state: AgentState) -> AgentState:
    title = state["selected_rule"]
    operator_keywords = state["operator_keywords"]
    tiebreak_keywords = state["tiebreak_keywords"]
    negative_keywords = state.get("negative_keywords", []) or []

    menu_mode = bool(state.get("menu_cafe_names"))
    kw_map = _build_kw_map(operator_keywords, tiebreak_keywords, negative_keywords)
    rule_str = _build_soar_rule(
        operator_keywords, tiebreak_keywords, kw_map,
        negative_keywords=negative_keywords, menu_mode=menu_mode,
    )
    cafe_items = _load_cafe_items(state.get("menu_cafe_names"))
    # trace는 필터 전(neg narrowing 단계까지 보여줌, XAI), Soar input은 부정 매장 제거 후
    tiebreak_trace = _trace_tiebreak(operator_keywords, tiebreak_keywords, negative_keywords, cafe_items)
    cafe_items = _filter_negative(cafe_items, operator_keywords, negative_keywords)

    kernel = sml.Kernel.CreateKernelInNewThread()
    agent = kernel.CreateAgent("cafe-recommender")
    input_link = agent.GetInputLink()

    for cafe_name, keywords in cafe_items.items():
        cafe_id = agent.CreateIdWME(input_link, "cafe")
        agent.CreateStringWME(cafe_id, "name", cafe_name)
        if menu_mode:
            agent.CreateStringWME(cafe_id, "has-menu", "true")
        for kw in keywords:
            if kw in kw_map:
                agent.CreateStringWME(cafe_id, "keyword", kw_map[kw])

    agent.Commit()
    agent.ExecuteCommandLine(rule_str)
    agent.RunSelfTilOutput()

    soar_result = _parse_output(agent)
    soar_result["trace"] = tiebreak_trace

    kernel.Shutdown()
    del kernel

    get_client().update_current_span(
        input={
            "title": title,
            "operator_keywords": operator_keywords,
            "tiebreak_keywords": tiebreak_keywords,
            "negative_keywords": negative_keywords,
        },
        output=soar_result,
    )
    return {"soar_result": soar_result}
