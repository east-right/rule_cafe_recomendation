import re
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe

from question_classifier.classify import classify_question
from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

KOREAN_THRESHOLD = 0.7


def _is_korean_enough(text: str) -> bool:
    non_ws = re.sub(r"\s", "", text)
    if not non_ws:
        return False
    korean = len(re.findall(r"[가-힣ᄀ-ᇿ㄰-㆏]", non_ws))
    return (korean / len(non_ws)) >= KOREAN_THRESHOLD


@observe()
def run(state: AgentState) -> AgentState:
    question = state["question"]

    if not _is_korean_enough(question):
        get_client().update_current_span(input=question, output="fallback")
        return {"question_type": "fallback"}

    label = classify_question(question)

    # invalid는 답변 불가 안내(fallback)로 처리
    if label == "invalid":
        label = "fallback"

    get_client().update_current_span(input=question, output=label)
    return {"question_type": label}
