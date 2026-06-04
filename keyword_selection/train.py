"""
sLLM 파인튜닝 스크립트 (Unsloth)
Usage: python train.py --config config/exaone_2.4b.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
from datasets import Dataset
from trl import SFTTrainer, TrainingArguments
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from prompt import SYSTEM_PROMPT, format_for_training

# ── 경로 설정 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train.jsonl"
TEST_PATH = ROOT / "data" / "test.jsonl"


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(path: Path) -> Dataset:
    with open(path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    formatted = [
        format_for_training(d["query"], d["candidates"], d["answer"])
        for d in data
    ]
    return Dataset.from_list(formatted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[INFO] 모델: {config['model_name']}")
    print(f"[INFO] 출력 경로: {config['output_dir']}")

    # ── 모델 로드 ─────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        dtype=None,  # auto
    )

    # chat template 적용
    tokenizer = get_chat_template(tokenizer, chat_template="auto")

    # ── LoRA 설정 ─────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── 데이터 로드 ───────────────────────────────────────
    print("[INFO] 데이터 로드 중...")
    train_dataset = load_dataset(TRAIN_PATH)
    test_dataset = load_dataset(TEST_PATH)
    print(f"  train: {len(train_dataset)}개 | test: {len(test_dataset)}개")

    # ── 학습 설정 ─────────────────────────────────────────
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type="cosine",
        fp16=not config["load_in_4bit"],
        bf16=False,
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=2,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        seed=42,
    )

    # ── 학습 ──────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": False},
        max_seq_length=config["max_seq_length"],
        args=training_args,
    )

    print("[INFO] 학습 시작!")
    trainer.train()

    # ── 저장 ──────────────────────────────────────────────
    print("[INFO] 모델 저장 중...")
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"[INFO] 저장 완료: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
