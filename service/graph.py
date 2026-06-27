from langgraph.graph import END, StateGraph

from service.nodes import impasse_resolve, menu_extract, question_valid, rule_select, soar_recommend
from service.state import AgentState

MAX_IMPASSE_ITER = 5

MENU_TYPES = ("menu-only", "menu-complex")


def _route_question(state: AgentState) -> str | list[str]:
    qt = state.get("question_type")
    if qt == "non-menu":
        return "rule_select"
    if qt in MENU_TYPES:
        return ["menu_extract", "rule_select"]
    return END


def _route_rule_select(state: AgentState) -> str:
    if state.get("question_type") in MENU_TYPES:
        return "soar_recommend"
    if state.get("selected_rule") == "none" or not state.get("selected_rule"):
        return END
    return "soar_recommend"


def _route_soar(state: AgentState) -> str:
    result = state.get("soar_result") or {}
    if (
        result.get("type") == "impasse"
        and state.get("impasse_iterations", 0) < MAX_IMPASSE_ITER
        and not state.get("impasse_exhausted")
    ):
        return "impasse_resolve"
    return END


def _route_impasse_resolve(state: AgentState) -> str:
    if state.get("impasse_exhausted"):
        return END
    return "soar_recommend"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("question_valid", question_valid.run)
    graph.add_node("menu_extract", menu_extract.run)
    graph.add_node("rule_select", rule_select.run)
    graph.add_node("soar_recommend", soar_recommend.run)
    graph.add_node("impasse_resolve", impasse_resolve.run)

    graph.set_entry_point("question_valid")

    graph.add_conditional_edges(
        "question_valid",
        _route_question,
        {"rule_select": "rule_select", "menu_extract": "menu_extract", END: END},
    )
    # menu_extract는 항상 soar_recommend로 (join 포인트)
    graph.add_edge("menu_extract", "soar_recommend")

    graph.add_conditional_edges(
        "rule_select",
        _route_rule_select,
        {"soar_recommend": "soar_recommend", END: END},
    )
    graph.add_conditional_edges(
        "soar_recommend",
        _route_soar,
        {"impasse_resolve": "impasse_resolve", END: END},
    )
    graph.add_conditional_edges(
        "impasse_resolve",
        _route_impasse_resolve,
        {"soar_recommend": "soar_recommend", END: END},
    )

    return graph.compile()
