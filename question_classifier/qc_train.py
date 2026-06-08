"""
질문 유효성 검증 모델 파인튜닝 스크립트 (RoBERTa 4-class 분류)
Usage: python train.py --config config/roberta_base.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ── 경로 설정 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train.jsonl"
TEST_PATH = ROOT / "data" / "test.jsonl"

# ── 라벨 매핑 ─────────────────────────────────────────────
ID2LABEL = {0: "non-menu", 1: "menu-only", 2: "menu-complex", 3: "invalid"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(path: Path) -> Dataset:
    with open(path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)


def tokenize(examples, tokenizer, max_length):
    return tokenizer(
        examples["question"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[INFO] 모델: {config['model_name']}")
    print(f"[INFO] 출력 경로: {config['output_dir']}")

    # ── 토크나이저 & 모델 로드 ────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── 데이터 로드 & 토크나이징 ──────────────────────────
    print("[INFO] 데이터 로드 중...")
    train_dataset = load_dataset(TRAIN_PATH)
    test_dataset = load_dataset(TEST_PATH)
    print(f"  train: {len(train_dataset)}개 | test: {len(test_dataset)}개")

    train_dataset = train_dataset.map(
        lambda x: tokenize(x, tokenizer, config["max_seq_length"]),
        batched=True,
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize(x, tokenizer, config["max_seq_length"]),
        batched=True,
    )

    # ── 학습 설정 ─────────────────────────────────────────
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=config["logging_steps"],
        report_to="none",
        seed=42,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("[INFO] 학습 시작!")
    trainer.train()

    # ── 저장 ──────────────────────────────────────────────
    print("[INFO] 모델 저장 중...")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[INFO] 저장 완료: {final_dir}")


if __name__ == "__main__":
    main()
