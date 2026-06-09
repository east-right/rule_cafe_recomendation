"""
menu-complex 질문 생성 스크립트
메뉴 키워드 + 비메뉴 키워드 조합으로 GPT가 질문 생성 (배치 처리)

Usage: python generate_menu_complex.py
"""

import json
import os
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from prompt import MENU_COMPLEX_SYSTEM, build_menu_complex_prompt

# ── 환경변수 로드 ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ── 경로 설정 ──────────────────────────────────────────────
MARKET_ITEM_PATH = ROOT / "data" / "market_item.csv"
VALID_QUESTIONS_PATH = ROOT / "data" / "valid_questions.json"
OUTPUT_PATH = ROOT / "question_classifier" / "data" / "menu_complex.jsonl"

# ── 상수 ──────────────────────────────────────────────────
TARGET_COUNT = 600
BATCH_SIZE = 10
SEED = 42
NON_MENU_SAMPLE_SIZE = 2


def load_menu_keywords() -> list[str]:
    df = pd.read_csv(MARKET_ITEM_PATH)
    # MENU 타입만 필터링
    menu_df = df[df['키워드타입'] == 'MENU']
    keywords = menu_df['최종_키워드'].dropna().apply(lambda x: x.split('_')[0]).unique().tolist()
    print(f"[INFO] 실제 메뉴 키워드: {len(keywords)}개")
    return keywords


def load_non_menu_keywords() -> list[str]:
    with open(VALID_QUESTIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    keywords = set()
    for d in data:
        if d['type'] in ['single', 'multi']:
            for kw in d['keywords']:
                keywords.add(kw['keyword'])

    keywords = list(keywords)
    print(f"[INFO] 비메뉴 키워드: {len(keywords)}개")
    return keywords


def generate_batch(
    client: OpenAI,
    batch: list[dict],
) -> list[str | None]:
    batch_user_prompt = ""
    for i, item in enumerate(batch, 1):
        batch_user_prompt += f"{i}. {build_menu_complex_prompt(item['menu'], item['non_menu'])}\n\n"

    batch_user_prompt += f"위 {len(batch)}개의 조건에 대해 각각 질문을 1개씩 생성하세요.\n반드시 번호와 함께 출력하세요. (예: 1. 질문내용)"

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": MENU_COMPLEX_SYSTEM},
                {"role": "user", "content": batch_user_prompt},
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
            for i in range(1, len(batch) + 1):
                if line.startswith(f"{i}."):
                    question = line[len(f"{i}."):].strip()
                    results.append(question)
                    break

        while len(results) < len(batch):
            results.append(None)

        return results[:len(batch)]

    except Exception as e:
        print(f"[ERROR] {e}")
        return [None] * len(batch)


def main():
    random.seed(SEED)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    menu_keywords = load_menu_keywords()
    non_menu_keywords = load_non_menu_keywords()

    batches = []
    batch = []
    for _ in range(TARGET_COUNT):
        item = {
            "menu": random.choice(menu_keywords),
            "non_menu": random.sample(non_menu_keywords, min(NON_MENU_SAMPLE_SIZE, len(non_menu_keywords))),
        }
        batch.append(item)
        if len(batch) == BATCH_SIZE:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    generated = []
    print(f"[INFO] 총 {len(batches)}개 배치 처리 시작 (배치당 {BATCH_SIZE}개)...")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for batch_items in tqdm(batches):
            questions = generate_batch(client, batch_items)
            for item, question in zip(batch_items, questions):
                if question:
                    record = {
                        "question": question,
                        "type": "menu-complex",
                        "menu_keyword": item["menu"],
                        "non_menu_keywords": item["non_menu"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    generated.append(record)

    print(f"\n[INFO] 생성 완료: {len(generated)}개")
    print(f"[INFO] 저장 경로: {OUTPUT_PATH}")
    print(f"\n샘플 3개:")
    for r in generated[:3]:
        print(f"  질문: {r['question']}")
        print(f"  메뉴: {r['menu_keyword']} | 비메뉴: {r['non_menu_keywords']}")
        print()


if __name__ == "__main__":
    main()