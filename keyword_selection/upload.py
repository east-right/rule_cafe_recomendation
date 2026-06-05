"""
파인튜닝된 모델 + 데이터 허깅페이스 업로드 스크립트
Usage: python upload.py --config config/qwen_1.5b.yaml
"""

import unsloth  # noqa: F401 - must be imported first

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
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # ── 모델 업로드 ───────────────────────────────────────
    print(f"[INFO] 모델 업로드 중: {hf_repo}")
    model.push_to_hub(hf_repo, token=HF_TOKEN, private=True)
    tokenizer.push_to_hub(hf_repo, token=HF_TOKEN, private=True)
    print("[INFO] 모델 업로드 완료")

    # ── 데이터 업로드 ─────────────────────────────────────
    api = HfApi(token=HF_TOKEN)
    data_dir = ROOT / "data"

    for filename in ["train.jsonl", "test.jsonl", "finetune_keyword_selection.jsonl"]:
        file_path = data_dir / filename
        if file_path.exists():
            print(f"[INFO] 데이터 업로드 중: {filename}")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=f"data/{filename}",
                repo_id=hf_repo,
                repo_type="model",
                token=HF_TOKEN,
            )
            print(f"[INFO] {filename} 업로드 완료")

    print(f"\n[INFO] 전체 업로드 완료: https://huggingface.co/{hf_repo}")


if __name__ == "__main__":
    main()