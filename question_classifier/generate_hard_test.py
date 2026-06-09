"""
Hard test 데이터 생성 스크립트 (GPT 배치 생성)
분류하기 어려운 엣지 케이스 300개 생성

클래스별 75개씩:
- non-menu: 줄임말, 모호한 표현
- menu-only: 줄임말, 신조어, 메뉴+맛집
- menu-complex: 메뉴가 암묵적으로 포함
- invalid: 카페 관련이지만 추천 불가, 경계 애매

Usage: python generate_hard_test.py
"""

import json
import os
import re
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from prompt import HARD_TEST_SYSTEM, build_hard_test_prompt

# ── 환경변수 로드 ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ── 경로 설정 ──────────────────────────────────────────────
OUTPUT_PATH = ROOT / "question_classifier" / "data" / "hard_test.jsonl"

# ── 상수 ──────────────────────────────────────────────────
PER_CLASS = 75
BATCH_SIZE = 15
SEED = 42

LABEL_MAP = {
    "non-menu": 0,
    "menu-only": 1,
    "menu-complex": 2,
    "invalid": 3,
}


def generate_batch(client: OpenAI, class_name: str, count: int) -> list[str]:
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": HARD_TEST_SYSTEM},
                {"role": "user", "content": build_hard_test_prompt(class_name, count)},
            ],
            max_completion_tokens=1000,
            temperature=0.9,
        )
        content = response.choices[0].message.content.strip()

        results = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            # "1. 질문" 형식 파싱
            match = re.match(r'^\d+\.\s*(.+)', line)
            if match:
                results.append(match.group(1).strip())

        return results

    except Exception as e:
        print(f"[ERROR] {e}")
        return []


def main():
    random.seed(SEED)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_generated = []
    n_batches = PER_CLASS // BATCH_SIZE  # 5배치 x 15개 = 75개

    print(f"[INFO] 클래스별 {PER_CLASS}개, 총 {PER_CLASS * 4}개 생성 시작...")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for class_name, label in LABEL_MAP.items():
            print(f"\n[INFO] {class_name} 생성 중...")
            class_generated = []

            for _ in tqdm(range(n_batches)):
                questions = generate_batch(client, class_name, BATCH_SIZE)
                for q in questions:
                    record = {
                        "question": q,
                        "label": label,
                        "hard": True,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    class_generated.append(record)

            print(f"  생성 완료: {len(class_generated)}개")
            all_generated.extend(class_generated)

    print(f"\n[INFO] 전체 생성 완료: {len(all_generated)}개")
    print(f"[INFO] 저장 경로: {OUTPUT_PATH}")

    print(f"\n클래스별 샘플:")
    for class_name, label in LABEL_MAP.items():
        samples = [d for d in all_generated if d["label"] == label][:2]
        print(f"\n  [{class_name}]")
        for s in samples:
            print(f"    - {s['question']}")


if __name__ == "__main__":
    main()
