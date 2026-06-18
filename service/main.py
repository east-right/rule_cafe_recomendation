import json
import sys

from langfuse import get_client

from service.graph import build_graph

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run(question: str) -> dict:
    lf = get_client()
    with lf.start_as_current_observation(
        name="cafe-recommend", as_type="agent", input=question
    ):
        result = get_graph().invoke({"question": question})
        lf.update_current_span(output=result.get("soar_result"))
    return result


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "조용하고 넓은 카페 추천해줘"
    result = run(question)
    print(json.dumps(result, ensure_ascii=False, indent=2))
