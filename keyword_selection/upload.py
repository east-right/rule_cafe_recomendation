"""
파인튜닝된 모델 허깅페이스 업로드 스크립트
Usage: python upload.py --config config/exaone_2.4b.yaml
"""

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ── 환경변수 로드 ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_WRITE")


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = ROOT / config["output_dir"] / "final"
    hf_repo = config["hf_repo"]

    print(f"[INFO] 모델 경로: {model_path}")
    print(f"[INFO] 업로드 대상: {hf_repo}")

    # ── 모델 로드 ─────────────────────────────────────────
    print("[INFO] 모델 로드 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="auto")

    # ── 허깅페이스 업로드 ─────────────────────────────────
    print(f"[INFO] 허깅페이스 업로드 중: {hf_repo}")
    model.push_to_hub(hf_repo, token=HF_TOKEN, private=True)
    tokenizer.push_to_hub(hf_repo, token=HF_TOKEN, private=True)

    print(f"[INFO] 업로드 완료: https://huggingface.co/{hf_repo}")


if __name__ == "__main__":
    main()
