"""
sLLM 키워드 선택 모델 평가 스크립트
Usage: python evaluate.py --config config/exaone_2.4b.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from prompt import SYSTEM_PROMPT, build_user_prompt

# ── 경로 설정 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
TEST_PATH = ROOT / "data" / "test.jsonl"


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_answer(
    query: str,
    candidates: list[dict],
    model,
    tokenizer,
    max_new_tokens: int = 32,
) -> str:
    user_prompt = build_user_prompt(query, candidates)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.1,
    )

    # 입력 부분 제거 후 디코딩
    generated = output_ids[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = ROOT / config["output_dir"] / "final"
    print(f"[INFO] 모델 로드: {model_path}")

    # ── 모델 로드 ─────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="auto")
    FastLanguageModel.for_inference(model)

    # ── 데이터 로드 ───────────────────────────────────────
    with open(TEST_PATH, encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]
    print(f"[INFO] test 데이터: {len(test_data)}개")

    # ── 평가 ──────────────────────────────────────────────
    total = 0
    correct = 0
    none_total = 0
    none_correct = 0
    match_total = 0
    match_correct = 0

    results = []

    print("[INFO] 평가 중...")
    for d in tqdm(test_data):
        pred = generate_answer(d["query"], d["candidates"], model, tokenizer)
        label = d["answer"]
        is_correct = pred == label

        total += 1
        if is_correct:
            correct += 1

        if label == "none":
            none_total += 1
            if is_correct:
                none_correct += 1
        else:
            match_total += 1
            if is_correct:
                match_correct += 1

        results.append({
            "query": d["query"],
            "answer": label,
            "pred": pred,
            "correct": is_correct,
        })

    # ── 결과 출력 ─────────────────────────────────────────
    accuracy = correct / total
    none_accuracy = none_correct / none_total if none_total > 0 else 0
    match_accuracy = match_correct / match_total if match_total > 0 else 0

    print(f"\n{'='*50}")
    print(f"모델: {config['model_name']}")
    print(f"{'='*50}")
    print(f"전체 Accuracy:  {accuracy:.4f} ({correct}/{total})")
    print(f"match Accuracy: {match_accuracy:.4f} ({match_correct}/{match_total})")
    print(f"none Accuracy:  {none_accuracy:.4f} ({none_correct}/{none_total})")
    print(f"{'='*50}")

    # ── 오답 샘플 출력 ────────────────────────────────────
    wrong = [r for r in results if not r["correct"]]
    print(f"\n오답 샘플 (최대 5개):")
    for r in wrong[:5]:
        print(f"  query:  {r['query']}")
        print(f"  answer: {r['answer']} | pred: {r['pred']}")
        print()

    # ── 결과 저장 ─────────────────────────────────────────
    output_dir = ROOT / config["output_dir"]
    result_path = output_dir / "eval_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": config["model_name"],
            "accuracy": accuracy,
            "match_accuracy": match_accuracy,
            "none_accuracy": none_accuracy,
            "total": total,
            "correct": correct,
            "details": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 결과 저장: {result_path}")


if __name__ == "__main__":
    main()
