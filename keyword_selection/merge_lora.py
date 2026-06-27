"""
LoRA adapter를 base 모델에 병합 → 단일 HF 모델로 저장.
이후 gguf 변환(llama.cpp)의 입력이 된다.

Usage: python keyword_selection/merge_lora.py
출력: keyword_selection/merged/  (HF 형식 병합 모델)
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_READ")
SLLM_REPO = "east-right/cafe-keyword-selection-qwen-1.5b"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT_DIR = ROOT / "keyword_selection" / "merged"


def _download_and_fix_adapter() -> str:
    """Unsloth 저장 어댑터 키 구조를 표준 PEFT로 변환 (rule_select와 동일)."""
    config_path = hf_hub_download(SLLM_REPO, "adapter_config.json", token=HF_TOKEN)
    weights_path = hf_hub_download(SLLM_REPO, "adapter_model.safetensors", token=HF_TOKEN)
    weights = load_file(weights_path)
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


def main():
    print("[INFO] adapter 다운로드/변환...")
    adapter_dir = _download_and_fix_adapter()

    print("[INFO] base 모델 로드...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)

    print("[INFO] adapter 병합...")
    model = PeftModel.from_pretrained(base, adapter_dir)
    merged = model.merge_and_unload()

    print(f"[INFO] 저장 → {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(OUT_DIR)
    AutoTokenizer.from_pretrained(SLLM_REPO, token=HF_TOKEN).save_pretrained(OUT_DIR)
    print("[INFO] 병합 완료")


if __name__ == "__main__":
    main()
