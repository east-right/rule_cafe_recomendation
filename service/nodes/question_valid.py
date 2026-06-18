import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langfuse.decorators import langfuse_context, observe
from transformers import pipeline

from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_READ")
MODEL_REPO = "east-right/cafe-question-classifier-koelectra-base"
KOREAN_THRESHOLD = 0.7

_LABEL_MAP = {
    "LABEL_0": "non-menu",
    "LABEL_1": "menu-only",
    "LABEL_2": "menu-complex",
    "LABEL_3": "invalid",
}

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model=MODEL_REPO,
            token=HF_TOKEN,
            device=-1,
        )
    return _classifier


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
        langfuse_context.update_current_observation(input=question, output="fallback")
        return {"question_type": "fallback"}

    result = _get_classifier()(question)[0]
    label = _LABEL_MAP.get(result["label"], result["label"])

    if label == "invalid":
        label = "fallback"

    langfuse_context.update_current_observation(
        input=question,
        output=label,
        metadata={"score": result["score"]},
    )
    return {"question_type": label}
