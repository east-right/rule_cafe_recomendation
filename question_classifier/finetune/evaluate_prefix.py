"""
prefix 추가 시 hard test 성능 변화 확인
"카페 추천해주는데 " + 질문 형태로 평가

Usage: python evaluate_prefix.py --config config/roberta_base.yaml
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
HARD_TEST_PATH = ROOT.parent / "data_augment" / "data" / "hard_test.jsonl"

ID2LABEL = {0: "non-menu", 1: "menu-only", 2: "menu-complex", 3: "invalid"}
PREFIX = "카페 추천 원해! 안 원할 수도 있는데 "


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict(text: str, model, tokenizer, device, max_length: int) -> str:
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=-1).item()
    return ID2LABEL[pred_id]


def evaluate(data: list[dict], model, tokenizer, device, max_length: int, use_prefix: bool) -> dict:
    labels = []
    preds = []
    results = []

    for d in tqdm(data):
        question = (PREFIX + d["question"]) if use_prefix else d["question"]
        pred_label = predict(question, model, tokenizer, device, max_length)
        true_label = ID2LABEL[d["label"]]

        labels.append(d["label"])
        preds.append(list(ID2LABEL.values()).index(pred_label))
        results.append({
            "original": d["question"],
            "question": question,
            "answer": true_label,
            "pred": pred_label,
            "correct": pred_label == true_label,
        })

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": accuracy, "f1_macro": f1, "results": results, "labels": labels, "preds": preds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = ROOT / config["output_dir"] / "final"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    with open(HARD_TEST_PATH, encoding="utf-8") as f:
        hard_data = [json.loads(line) for line in f]

    print(f"[INFO] Hard test: {len(hard_data)}개\n")

    # prefix 없이
    print("평가 중 (prefix 없음)...")
    no_prefix = evaluate(hard_data, model, tokenizer, device, config["max_seq_length"], use_prefix=False)

    # prefix 있음
    print("평가 중 (prefix 있음)...")
    with_prefix = evaluate(hard_data, model, tokenizer, device, config["max_seq_length"], use_prefix=True)

    # ── 결과 출력 ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Hard Test: prefix 없음 vs prefix 있음 비교")
    print(f"{'='*60}")
    print(f"{'':25} {'prefix 없음':>15} {'prefix 있음':>15}")
    print(f"{'-'*60}")
    print(f"{'Accuracy':25} {no_prefix['accuracy']:>15.4f} {with_prefix['accuracy']:>15.4f}")
    print(f"{'F1 (macro)':25} {no_prefix['f1_macro']:>15.4f} {with_prefix['f1_macro']:>15.4f}")
    print(f"{'='*60}")

    # 클래스별 비교
    print(f"\n[prefix 없음] 클래스별 리포트:")
    print(classification_report(no_prefix["labels"], no_prefix["preds"], target_names=list(ID2LABEL.values())))

    print(f"\n[prefix 있음] 클래스별 리포트:")
    print(classification_report(with_prefix["labels"], with_prefix["preds"], target_names=list(ID2LABEL.values())))

    # prefix로 바뀐 케이스
    changed = [
        r for r1, r2 in zip(no_prefix["results"], with_prefix["results"])
        if r1["pred"] != r2["pred"]
        for r in [{"original": r1["original"], "answer": r1["answer"],
                   "before": r1["pred"], "after": r2["pred"]}]
    ]
    print(f"\nprefix로 인해 바뀐 예측: {len(changed)}개")
    print(f"\n샘플 10개:")
    for r in changed[:10]:
        mark = "✅" if r["after"] == r["answer"] else "❌"
        print(f"  {mark} [{r['answer']}] {r['original']}")
        print(f"     {r['before']} → {r['after']}")
        print()


if __name__ == "__main__":
    main()
