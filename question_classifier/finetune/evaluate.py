"""
질문 유효성 검증 모델 평가 스크립트
Usage: python evaluate.py --config config/roberta_base.yaml
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
TEST_PATH = ROOT.parent / "data_augment" / "data" / "test.jsonl"
HARD_TEST_PATH = ROOT.parent / "data_augment" / "data" / "hard_test.jsonl"

ID2LABEL = {0: "non-menu", 1: "menu-only", 2: "menu-complex", 3: "invalid"}


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def predict(text: str, model, tokenizer, device, max_length: int) -> tuple[str, float]:
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed = time.time() - start

    pred_id = torch.argmax(outputs.logits, dim=-1).item()
    return ID2LABEL[pred_id], elapsed


def evaluate_dataset(
    data: list[dict],
    model,
    tokenizer,
    device,
    max_length: int,
    dataset_name: str,
) -> dict:
    labels = []
    preds = []
    inference_times = []
    results = []

    for d in tqdm(data, desc=dataset_name):
        pred_label, elapsed = predict(d["question"], model, tokenizer, device, max_length)
        true_label = ID2LABEL[d["label"]]

        labels.append(d["label"])
        preds.append(list(ID2LABEL.values()).index(pred_label))
        inference_times.append(elapsed)

        results.append({
            "question": d["question"],
            "answer": true_label,
            "pred": pred_label,
            "correct": pred_label == true_label,
            "inference_time": round(elapsed, 4),
        })

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    avg_time = sum(inference_times) / len(inference_times)

    print(f"\n{'='*55}")
    print(f"{dataset_name} 평가 결과")
    print(f"{'='*55}")
    print(f"전체 Accuracy: {accuracy:.4f}")
    print(f"F1 (macro):    {f1:.4f}")
    print(f"{'='*55}")
    print(f"평균 추론 시간: {avg_time:.4f}s / 건")
    print(f"{'='*55}")
    print(f"\n클래스별 리포트:")
    print(classification_report(labels, preds, target_names=list(ID2LABEL.values())))

    wrong = [r for r in results if not r["correct"]]
    print(f"오답 샘플 (최대 5개):")
    for r in wrong[:5]:
        print(f"  question: {r['question']}")
        print(f"  answer: {r['answer']} | pred: {r['pred']}")
        print()

    return {
        "accuracy": accuracy,
        "f1_macro": f1,
        "avg_inference_time": round(avg_time, 4),
        "total": len(data),
        "correct": sum(1 for r in results if r["correct"]),
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = ROOT / config["output_dir"] / "final"
    print(f"[INFO] 모델 로드: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 디바이스: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    # ── 일반 test 평가 ────────────────────────────────────
    test_data = load_jsonl(TEST_PATH)
    print(f"\n[INFO] 일반 test: {len(test_data)}개")
    test_results = evaluate_dataset(
        test_data, model, tokenizer, device,
        config["max_seq_length"], "일반 Test"
    )

    # ── Hard test 평가 ────────────────────────────────────
    hard_results = None
    if HARD_TEST_PATH.exists():
        hard_data = load_jsonl(HARD_TEST_PATH)
        print(f"\n[INFO] Hard test: {len(hard_data)}개")
        hard_results = evaluate_dataset(
            hard_data, model, tokenizer, device,
            config["max_seq_length"], "Hard Test"
        )

        # ── 비교 요약 ─────────────────────────────────────
        print(f"\n{'='*55}")
        print(f"일반 vs Hard Test 비교")
        print(f"{'='*55}")
        print(f"{'':20} {'일반 Test':>15} {'Hard Test':>15}")
        print(f"{'-'*55}")
        print(f"{'Accuracy':20} {test_results['accuracy']:>15.4f} {hard_results['accuracy']:>15.4f}")
        print(f"{'F1 (macro)':20} {test_results['f1_macro']:>15.4f} {hard_results['f1_macro']:>15.4f}")
        print(f"{'='*55}")

    # ── 결과 저장 ─────────────────────────────────────────
    output_dir = ROOT / config["output_dir"]
    result_path = output_dir / "eval_results.json"
    save_data = {
        "model": config["model_name"],
        "test": test_results,
    }
    if hard_results:
        save_data["hard_test"] = hard_results

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] 결과 저장: {result_path}")


if __name__ == "__main__":
    main()