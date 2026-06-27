import json
import os
import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from langfuse import get_client as langfuse_client, observe

from keyword_selection.search import INDEX_NAME, get_client, load_model, search
from service.prompt import RULE_SELECT_SYSTEM, build_rule_select_user
from service.state import AgentState

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_READ")
SLLM_REPO = "east-right/cafe-keyword-selection-qwen-1.5b"
TOP_K = 10

_embedding_model = None
_os_client = None
_sllm_model = None
_sllm_tokenizer = None


def _get_retriever():
    global _embedding_model, _os_client
    if _embedding_model is None:
        _embedding_model = load_model()
    if _os_client is None:
        _os_client = get_client()
    return _embedding_model, _os_client


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _download_and_fix_adapter() -> str:
    """Unsloth 저장 어댑터의 키 구조를 표준 PEFT로 변환해 임시 디렉토리에 저장."""
    config_path = hf_hub_download(SLLM_REPO, "adapter_config.json", token=HF_TOKEN)
    weights_path = hf_hub_download(SLLM_REPO, "adapter_model.safetensors", token=HF_TOKEN)

    weights = load_file(weights_path)
    # Unsloth: base_model.model.model.model.<path>
    # PEFT 표준: base_model.model.<path>
    fixed = {
        k.replace("base_model.model.model.model.", "base_model.model."): v
        for k, v in weights.items()
    }

    tmp = tempfile.mkdtemp()
    save_file(fixed, os.path.join(tmp, "adapter_model.safetensors"))
    with open(config_path) as f:
        config = json.load(f)
    with open(os.path.join(tmp, "adapter_config.json"), "w") as f:
        json.dump(config, f)

    return tmp


def _get_sllm():
    global _sllm_model, _sllm_tokenizer
    if _sllm_model is None:
        _sllm_tokenizer = AutoTokenizer.from_pretrained(SLLM_REPO, token=HF_TOKEN)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
        )
        adapter_dir = _download_and_fix_adapter()
        _sllm_model = PeftModel.from_pretrained(base, adapter_dir)
        _sllm_model = _sllm_model.to("cpu")
        _sllm_model.eval()
    return _sllm_model, _sllm_tokenizer


def _fetch_keywords_from_os(title: str) -> dict:
    _, client = _get_retriever()
    res = client.search(
        index=INDEX_NAME,
        body={
            "query": {"term": {"title": title}},
            "_source": ["operator_keywords", "tiebreak_keywords"],
        },
    )
    hits = res["hits"]["hits"]
    return hits[0]["_source"] if hits else {}


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
        {"role": "system", "content": RULE_SELECT_SYSTEM},
        {"role": "user", "content": build_rule_select_user(question, candidates)},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cpu")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            repetition_penalty=1.1,
        )
    generated = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@observe()
def run(state: AgentState) -> AgentState:
    question = state["question"]

    candidates = _retrieve(question)
    selected = _select_rule(question, candidates)

    if selected == "none":
        langfuse_client().update_current_span(
            input=question,
            output="none",
            metadata={"candidates": candidates},
        )
        return {
            "rule_candidates": candidates,
            "selected_rule": "none",
            "operator_keywords": [],
            "tiebreak_keywords": [],
            "negative_keywords": [],
        }

    keywords = _fetch_keywords_from_os(selected)
    langfuse_client().update_current_span(
        input=question,
        output=selected,
        metadata={
            "candidates": candidates,
            "operator_keywords": keywords.get("operator_keywords", []),
            "tiebreak_keywords": keywords.get("tiebreak_keywords", []),
        },
    )
    return {
        "rule_candidates": candidates,
        "selected_rule": selected,
        "operator_keywords": keywords.get("operator_keywords", []),
        "tiebreak_keywords": keywords.get("tiebreak_keywords", []),
        "negative_keywords": keywords.get("negative_keywords", []),
    }
