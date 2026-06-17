from langgraph.graph import END, StateGraph

from service.nodes import question_valid, rule_select, soar_recommend
from service.state import AgentState


def _route_question(state: AgentState) -> str:
    qt = state.get("question_type")
    if qt == "non-menu":
        return "rule_select"
    # fallback / menu-only / menu-complex → 미구현, END
    return END


def _route_rule_select(state: AgentState) -> str:
    if state.get("selected_rule") == "none" or not state.get("selected_rule"):
        return END
    return "soar_recommend"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("question_valid", question_valid.run)
    graph.add_node("rule_select", rule_select.run)
    graph.add_node("soar_recommend", soar_recommend.run)

    graph.set_entry_point("question_valid")

    graph.add_conditional_edges(
        "question_valid",
        _route_question,
        {"rule_select": "rule_select", END: END},
    )
    graph.add_conditional_edges(
        "rule_select",
        _route_rule_select,
        {"soar_recommend": "soar_recommend", END: END},
    )
    graph.add_edge("soar_recommend", END)

    return graph.compile()
