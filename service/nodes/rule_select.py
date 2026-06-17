import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from keyword_selection.search import get_client, load_model, search
from keyword_selection.prompt import SYSTEM_PROMPT, build_user_prompt
from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_READ")
SLLM_REPO = "east-right/cafe-keyword-selection-qwen-1.5b"
SOAR_KEYWORDS_PATH = ROOT / "data" / "soar_rule_keywords.json"
TOP_K = 10

_embedding_model = None
_os_client = None
_sllm_model = None
_sllm_tokenizer = None
_rule_keywords: dict[str, dict] = {}


def _get_retriever():
    global _embedding_model, _os_client
    if _embedding_model is None:
        _embedding_model = load_model()
    if _os_client is None:
        _os_client = get_client()
    return _embedding_model, _os_client


def _get_sllm():
    global _sllm_model, _sllm_tokenizer
    if _sllm_model is None:
        _sllm_tokenizer = AutoTokenizer.from_pretrained(SLLM_REPO, token=HF_TOKEN)
        _sllm_model = AutoModelForCausalLM.from_pretrained(
            SLLM_REPO,
            token=HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    return _sllm_model, _sllm_tokenizer


def _load_rule_keywords() -> dict[str, dict]:
    global _rule_keywords
    if not _rule_keywords:
        with open(SOAR_KEYWORDS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _rule_keywords = {r["title"]: r for r in data}
    return _rule_keywords


def _retrieve(question: str) -> list[dict]:
    model, client = _get_retriever()
    results = search(question, client, model, TOP_K)
    return [
        {"rank": i + 1, "title": r["title"], "description": r["description"]}
        for i, r in enumerate(results)
    ]


def _select_rule(question: str, candidates: list[dict]) -> str:
    model, tokenizer = _get_sllm()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, candidates)},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            repetition_penalty=1.1,
        )
    generated = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run(state: AgentState) -> AgentState:
    question = state["question"]

    candidates = _retrieve(question)
    selected = _select_rule(question, candidates)

    if selected == "none":
        return {
            "rule_candidates": candidates,
            "selected_rule": "none",
            "operator_keywords": [],
            "tiebreak_keywords": [],
        }

    keywords = _load_rule_keywords().get(selected, {})
    return {
        "rule_candidates": candidates,
        "selected_rule": selected,
        "operator_keywords": keywords.get("operator_keywords", []),
        "tiebreak_keywords": keywords.get("tiebreak_keywords", []),
    }
