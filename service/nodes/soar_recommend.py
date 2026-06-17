import os
import sqlite3
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SOAR_HOME = os.getenv("SOAR_HOME", str(ROOT / "soar" / "sml" / "Soar" / "out"))
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


def _make_operator(title: str, operator_keywords: list[str]) -> str:
    kw_list = " ".join([f"|{kw}|" for kw in operator_keywords])
    return f"""sp {{recommend*OPERATOR*{title}
   (state <s> ^io.input-link <il>)
   (<il> ^cafe <c>)
   (<c> ^name <c-name> ^keyword << {kw_list} >>)
-->
   (<s> ^operator <o> +)
   (<o> ^name recommend-cafe ^cafe-name <c-name>)
}}
"""


def _make_s1_judge(title: str, keyword: str) -> str:
    return f"""sp {{recommend*S1*judge*{title}
   (state <s> ^operator <o1> +
              ^operator <o2> +
              ^io.input-link <il>)
   (<o1> ^name recommend-cafe ^cafe-name <c1-name>)
   (<o2> ^name recommend-cafe ^cafe-name <c2-name> <> <c1-name>)
   (<il> ^cafe <c1> ^cafe <c2>)
   (<c1> ^name <c1-name> ^keyword |{keyword}|)
   (<c2> ^name <c2-name>)
   - {{ (<c2> ^keyword |{keyword}|) }}
-->
   (<s> ^operator <o1> > <o2>)
}}
"""


def _make_sn_judge(title: str, keyword: str, depth: int) -> str:
    chain = "".join(
        f"   (<s{i}> ^superstate nil)\n" if i == 1
        else f"   (<s{i}> ^superstate <s{i-1}>)\n"
        for i in range(1, depth + 1)
    )
    return f"""sp {{resolve*tie*S{depth}*judge*{title}
   (state <s> ^impasse <any-impasse>
              ^superstate <s{depth}>
              ^item <o1> ^item <o2>
              ^top-state <ts>)
{chain}   (<o1> ^name recommend-cafe ^cafe-name <c1-name>)
   (<o2> ^name recommend-cafe ^cafe-name <c2-name> <> <c1-name>)
   (<ts> ^io.input-link <il>)
   (<il> ^cafe <c1> ^cafe <c2>)
   (<c1> ^name <c1-name> ^keyword |{keyword}|)
   (<c2> ^name <c2-name>)
   - {{ (<c2> ^keyword |{keyword}|) }}
-->
   (<s{depth-1}> ^operator <o1> > <o2>)
}}
"""


def _make_llm_fallback(title: str, depth: int) -> str:
    chain = "".join(
        f"   (<s{i}> ^superstate nil)\n" if i == 1
        else f"   (<s{i}> ^superstate <s{i-1}>)\n"
        for i in range(1, depth + 1)
    )
    return f"""sp {{resolve*tie*S{depth}*ask-llm*{title}
   (state <s> ^impasse <any-impasse>
              ^superstate <s{depth}>
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


def _build_soar_rule(title: str, operator_keywords: list[str], tiebreak_keywords: list[str]) -> str:
    safe_title = title.replace(" ", "_")
    parts = [_BOILERPLATE, _make_operator(safe_title, operator_keywords)]

    if tiebreak_keywords:
        parts.append(_make_s1_judge(safe_title, tiebreak_keywords[0]))
        for i, kw in enumerate(tiebreak_keywords[1:], start=2):
            parts.append(_make_sn_judge(safe_title, kw, i))
        llm_depth = len(tiebreak_keywords) + 1
    else:
        llm_depth = 1

    parts.append(_make_llm_fallback(safe_title, llm_depth))
    return "\n".join(parts)


# ── SQLite 로드 ────────────────────────────────────────────────

def _load_cafe_items() -> dict[str, list[str]]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT cafe_name, keyword FROM cafe_keywords WHERE sentiment = '긍정'"
    ).fetchall()
    conn.close()

    result: dict[str, list[str]] = {}
    for row in rows:
        result.setdefault(row["cafe_name"], []).append(row["keyword"])
    return result


# ── output-link 파싱 ───────────────────────────────────────────

def _parse_output(output_link) -> dict:
    result = {"type": "no_output", "cafe": None, "candidates": []}
    n = output_link.GetNumberChildren()
    for i in range(n):
        wme = output_link.GetChild(i)
        attr = wme.GetAttribute()
        if attr == "final-recommendation":
            result["type"] = "recommendation"
            result["cafe"] = wme.GetValueAsString()
        elif attr == "ask-llm":
            result["type"] = "impasse"
            if wme.IsIdentifier():
                req_id = wme.ConvertToIdentifier()
                for j in range(req_id.GetNumberChildren()):
                    req_wme = req_id.GetChild(j)
                    if req_wme.GetAttribute() in ("cand1", "cand2"):
                        result["candidates"].append(req_wme.GetValueAsString())
    return result


# ── 노드 진입점 ────────────────────────────────────────────────

def run(state: AgentState) -> AgentState:
    title = state["selected_rule"]
    operator_keywords = state["operator_keywords"]
    tiebreak_keywords = state["tiebreak_keywords"]

    rule_str = _build_soar_rule(title, operator_keywords, tiebreak_keywords)
    cafe_items = _load_cafe_items()

    kernel = sml.Kernel.CreateKernelInNewThread()
    agent = kernel.CreateAgent("cafe-recommender")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".soar", delete=False, encoding="utf-8"
    ) as f:
        f.write(rule_str)
        temp_path = f.name

    try:
        agent.LoadProductions(temp_path)
    finally:
        os.unlink(temp_path)

    input_link = agent.GetInputLink()
    for cafe_name, keywords in cafe_items.items():
        cafe_id = input_link.CreateIdWME("cafe")
        cafe_id.CreateStringWME("name", cafe_name)
        for kw in keywords:
            cafe_id.CreateStringWME("keyword", kw)

    agent.Commit()
    agent.RunSelfTilOutput()

    soar_result = _parse_output(agent.GetOutputLink())

    kernel.Shutdown()
    del kernel

    return {"soar_result": soar_result}
