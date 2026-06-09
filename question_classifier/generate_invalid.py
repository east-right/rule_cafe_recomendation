"""
invalid 질문 생성 스크립트 (GPT 배치 생성)
카페 추천 시스템이 처리할 수 없는 질문 600개 생성

유형 비율:
- 유형1 (카페 외 업종 추천): 40%
- 유형2 (카페 관련이지만 추천 불가): 35%
- 유형3 (카페와 완전 무관): 25%

Usage: python generate_invalid.py
"""

import json
import os
import re
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from prompt import INVALID_SYSTEM, build_invalid_prompt

# ── 환경변수 로드 ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ── 경로 설정 ──────────────────────────────────────────────
OUTPUT_PATH = ROOT / "question_classifier" / "data" / "invalid.jsonl"

# ── 상수 ──────────────────────────────────────────────────
TARGET_COUNT = 600
BATCH_SIZE = 15  # 배치당 생성 수 (유형1:6, 유형2:5, 유형3:4)
SEED = 42


def generate_batch(client: OpenAI, type1: int, type2: int, type3: int) -> list[dict]:
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": INVALID_SYSTEM},
                {"role": "user", "content": build_invalid_prompt(type1, type2, type3)},
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
            # [유형1], [유형2], [유형3] 파싱
            match = re.match(r'\[유형(\d)\]\s*(.+)', line)
            if match:
                type_num = int(match.group(1))
                question = match.group(2).strip()
                results.append({
                    "question": question,
                    "type": "invalid",
                    "invalid_type": type_num,
                })

        return results

    except Exception as e:
        print(f"[ERROR] {e}")
        return []


def main():
    random.seed(SEED)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    generated = []
    # 배치당 유형1:6, 유형2:5, 유형3:4 = 15개
    n_batches = TARGET_COUNT // BATCH_SIZE

    print(f"[INFO] 총 {n_batches}개 배치 처리 시작 (배치당 {BATCH_SIZE}개)...")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for _ in tqdm(range(n_batches)):
            results = generate_batch(client, type1=6, type2=5, type3=4)
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                generated.append(r)

    from collections import Counter
    type_dist = Counter(r["invalid_type"] for r in generated)

    print(f"\n[INFO] 생성 완료: {len(generated)}개")
    print(f"  유형1 (카페 외 업종): {type_dist[1]}개")
    print(f"  유형2 (추천 불가):    {type_dist[2]}개")
    print(f"  유형3 (완전 무관):    {type_dist[3]}개")
    print(f"\n샘플 3개:")
    for r in generated[:3]:
        print(f"  [유형{r['invalid_type']}] {r['question']}")


if __name__ == "__main__":
    main()