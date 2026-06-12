"""
질문 유효성 검증 모델 HuggingFace 업로드
Usage: python upload.py --config config/roberta_base.yaml
"""

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── 환경변수 로드 ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent.parent / ".env")

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

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

    print(f"[INFO] 업로드 중: {hf_repo}")
    model.push_to_hub(hf_repo, token=HF_TOKEN, private=True)
    tokenizer.push_to_hub(hf_repo, token=HF_TOKEN, private=True)

    print(f"[INFO] 업로드 완료: https://huggingface.co/{hf_repo}")


if __name__ == "__main__":
    main()
