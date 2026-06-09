"""
학습 데이터 스타일 다양화 augmentation 스크립트
기존 질문을 구어체, 줄임말, 짧은 문장 등 다양한 스타일로 변환

Usage: python generate_augmented.py
"""

import json
import os
import re
import random
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from prompt import AUGMENT_SYSTEM, build_augment_prompt

# ── 환경변수 로드 ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ── 경로 설정 ──────────────────────────────────────────────
DATA_DIR = ROOT / "question_classifier" / "data_augment" / "data"

TRAINVAL_PATH = DATA_DIR / "trainval.jsonl"
OUTPUT_PATH = DATA_DIR / "augmented.jsonl"

# ── 상수 ──────────────────────────────────────────────────
BATCH_SIZE = 5        # 한번에 변환할 질문 수
COUNT_PER_Q = 3       # 질문당 변환 개수
SAMPLE_PER_CLASS = 80 # 클래스당 원본 샘플 수 (80 x 3 = 240개 augmentation)
SEED = 42

LABEL_MAP = {0: "non-menu", 1: "menu-only", 2: "menu-complex", 3: "invalid"}


def load_trainval() -> dict[int, list[dict]]:
    with open(TRAINVAL_PATH, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    by_class = defaultdict(list)
    for d in data:
        by_class[d["label"]].append(d)
    return by_class


def generate_batch(client: OpenAI, questions: list[str], count_per_q: int) -> dict[int, list[str]]:
    """배치로 여러 질문 변환. {원본인덱스: [변환질문들]} 반환"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": AUGMENT_SYSTEM},
                {"role": "user", "content": build_augment_prompt(questions, count_per_q)},
            ],
            max_completion_tokens=1500,
            temperature=0.9,
        )
        content = response.choices[0].message.content.strip()

        results = defaultdict(list)
        current_idx = None

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # [원본N] 태그 감지
            tag_match = re.match(r'\[원본(\d+)\]', line)
            if tag_match:
                current_idx = int(tag_match.group(1)) - 1
                continue

            # 번호 있는 질문 파싱
            q_match = re.match(r'^\d+\.\s*(.+)', line)
            if q_match and current_idx is not None:
                results[current_idx].append(q_match.group(1).strip())

        return results

    except Exception as e:
        print(f"[ERROR] {e}")
        return {}


def main():
    random.seed(SEED)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    by_class = load_trainval()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_generated = []

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for label, class_name in LABEL_MAP.items():
            samples = random.sample(by_class[label], min(SAMPLE_PER_CLASS, len(by_class[label])))
            print(f"\n[INFO] [{class_name}] {len(samples)}개 샘플 augmentation 시작...")

            # 배치로 처리
            for i in tqdm(range(0, len(samples), BATCH_SIZE)):
                batch = samples[i:i + BATCH_SIZE]
                questions = [d["question"] for d in batch]

                results = generate_batch(client, questions, COUNT_PER_Q)

                for q_idx, augmented_questions in results.items():
                    for aug_q in augmented_questions:
                        record = {
                            "question": aug_q,
                            "label": label,
                            "original": questions[q_idx],
                            "augmented": True,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        all_generated.append(record)

    # 결과 요약
    from collections import Counter
    dist = Counter(d["label"] for d in all_generated)
    print(f"\n[INFO] Augmentation 완료!")
    print(f"  총 생성: {len(all_generated)}개")
    for label, class_name in LABEL_MAP.items():
        print(f"  {class_name}: {dist[label]}개")

    print(f"\n클래스별 샘플:")
    for label, class_name in LABEL_MAP.items():
        samples = [d for d in all_generated if d["label"] == label][:2]
        print(f"\n  [{class_name}]")
        for s in samples:
            print(f"    원본: {s['original']}")
            print(f"    변환: {s['question']}")


if __name__ == "__main__":
    main()
