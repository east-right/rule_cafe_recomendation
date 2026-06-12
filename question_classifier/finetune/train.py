"""
질문 유효성 검증 모델 파인튜닝 스크립트 (RoBERTa 4-class, 5-fold 교차검증)
Usage: python train.py --config config/roberta_base.yaml
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ── 경로 설정 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
TRAINVAL_PATH = ROOT.parent / "data_augment" / "data" / "trainval.jsonl"

# ── 라벨 매핑 ─────────────────────────────────────────────
ID2LABEL = {0: "non-menu", 1: "menu-only", 2: "menu-complex", 3: "invalid"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

N_FOLDS = 5


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_dataset(data: list[dict], tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_list(data)
    dataset = dataset.map(
        lambda x: tokenizer(
            x["question"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
    )
    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def train_fold(
    fold: int,
    train_data: list[dict],
    val_data: list[dict],
    config: dict,
    tokenizer,
    output_dir: Path,
) -> dict:
    print(f"\n[Fold {fold}] train: {len(train_data)}개 | val: {len(val_data)}개")

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = make_dataset(train_data, tokenizer, config["max_seq_length"])
    val_dataset = make_dataset(val_data, tokenizer, config["max_seq_length"])

    fold_output_dir = output_dir / f"fold-{fold}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(fold_output_dir),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=config["logging_steps"],
        report_to="none",
        seed=42,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    # train/val loss 로그 수집
    log_history = trainer.state.log_history
    train_logs = [l for l in log_history if "loss" in l and "eval_loss" not in l]
    eval_logs = [l for l in log_history if "eval_loss" in l]

    print(f"[Fold {fold}] train_loss: {train_result.training_loss:.4f} | "
          f"val_loss: {eval_logs[-1]['eval_loss']:.4f} | "
          f"accuracy: {eval_metrics['eval_accuracy']:.4f} | "
          f"f1: {eval_metrics['eval_f1_macro']:.4f}")

    return {
        "fold": fold,
        "train_loss": train_result.training_loss,
        "val_loss": eval_logs[-1]["eval_loss"],
        "accuracy": eval_metrics["eval_accuracy"],
        "f1_macro": eval_metrics["eval_f1_macro"],
        "train_logs": train_logs,
        "eval_logs": eval_logs,
        "model": model,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[INFO] 모델: {config['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    print("[INFO] 데이터 로드 중...")
    all_data = load_data(TRAINVAL_PATH)
    labels = [d["label"] for d in all_data]
    print(f"  trainval: {len(all_data)}개")

    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    best_f1 = -1
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_data, labels), 1):
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]

        result = train_fold(fold, train_data, val_data, config, tokenizer, output_dir)
        fold_results.append(result)

        if result["f1_macro"] > best_f1:
            best_f1 = result["f1_macro"]
            best_model = result["model"]

    # ── 결과 출력 ─────────────────────────────────────────
    avg_acc = np.mean([r["accuracy"] for r in fold_results])
    avg_f1 = np.mean([r["f1_macro"] for r in fold_results])
    avg_train_loss = np.mean([r["train_loss"] for r in fold_results])
    avg_val_loss = np.mean([r["val_loss"] for r in fold_results])
    std_acc = np.std([r["accuracy"] for r in fold_results])
    std_f1 = np.std([r["f1_macro"] for r in fold_results])

    print(f"\n{'='*60}")
    print(f"5-Fold 교차검증 결과")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: train_loss={r['train_loss']:.4f} | "
              f"val_loss={r['val_loss']:.4f} | "
              f"accuracy={r['accuracy']:.4f} | "
              f"f1={r['f1_macro']:.4f}")
    print(f"{'='*60}")
    print(f"  평균 train_loss: {avg_train_loss:.4f}")
    print(f"  평균 val_loss:   {avg_val_loss:.4f}")
    print(f"  평균 accuracy:   {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"  평균 f1_macro:   {avg_f1:.4f} ± {std_f1:.4f}")

    if avg_train_loss < avg_val_loss * 0.7:
        print(f"\n  ⚠️  오버피팅 의심: train_loss({avg_train_loss:.4f}) << val_loss({avg_val_loss:.4f})")
    else:
        print(f"\n  ✅ 오버피팅 없음")
    print(f"{'='*60}")

    # ── best 모델 저장 ────────────────────────────────────
    print(f"\n[INFO] best 모델 저장 중 (f1: {best_f1:.4f})...")
    final_dir = output_dir / "final"
    best_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[INFO] 저장 완료: {final_dir}")

    # ── 결과 저장 ─────────────────────────────────────────
    with open(output_dir / "fold_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "avg_accuracy": avg_acc,
            "avg_f1_macro": avg_f1,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "std_accuracy": std_acc,
            "std_f1_macro": std_f1,
            "folds": [
                {
                    "fold": r["fold"],
                    "train_loss": r["train_loss"],
                    "val_loss": r["val_loss"],
                    "accuracy": r["accuracy"],
                    "f1_macro": r["f1_macro"],
                    "train_logs": r["train_logs"],
                    "eval_logs": r["eval_logs"],
                }
                for r in fold_results
            ],
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()