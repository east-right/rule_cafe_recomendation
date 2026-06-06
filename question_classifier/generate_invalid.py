"""
invalid 질문 추출 스크립트
KLUE-MRC에서 카페와 무관한 질문 600개 샘플링

Usage: python generate_invalid.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "question_classifier" / "data" / "invalid.jsonl"

# ── 상수 ──────────────────────────────────────────────────
TARGET_COUNT = 600
SEED = 42


def main():
    random.seed(SEED)

    print("[INFO] KLUE-MRC 데이터 로드 중...")
    ds = load_dataset("klue/klue", "mrc", split="train")
    print(f"[INFO] 전체 데이터: {len(ds)}개")

    # 질문만 추출 후 중복 제거
    questions = list(set(ds["question"]))
    print(f"[INFO] unique 질문: {len(questions)}개")

    # 랜덤 샘플링
    sampled = random.sample(questions, TARGET_COUNT)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] {TARGET_COUNT}개 저장 중...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for q in tqdm(sampled):
            record = {
                "question": q,
                "type": "invalid",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[INFO] 완료: {TARGET_COUNT}개")
    print(f"[INFO] 저장 경로: {OUTPUT_PATH}")
    print(f"\n샘플 3개:")
    for q in sampled[:3]:
        print(f"  {q}")


if __name__ == "__main__":
    main()
