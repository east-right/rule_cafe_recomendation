import json
import sys

from service.graph import build_graph

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run(question: str) -> dict:
    result = get_graph().invoke({"question": question})
    return result


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "조용하고 넓은 카페 추천해줘"
    result = run(question)
    print(json.dumps(result, ensure_ascii=False, indent=2))
