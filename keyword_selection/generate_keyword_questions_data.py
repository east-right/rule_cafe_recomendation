"""
sLLM 파인튜닝 질문 생성 스크립트 v2
rule title 기반 질문 생성 (구어체 + 일반체) + 메뉴-복합 증강

180개 rule x 20개 = 3600개 + 증강 20% = ~4320개

Usage: python generate_keyword_questions_data.py
"""

import json
import os
import re
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from prompt import QUESTION_GEN_SYSTEM, build_question_gen_prompt

# ── 환경변수 로드 ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ── 경로 설정 ──────────────────────────────────────────────
RULE_PATH = ROOT / "data" / "rule_metadata_merged.json"
OUTPUT_PATH = ROOT / "keyword_selection" / "data" / "keyword_questions.jsonl"

# ── 상수 ──────────────────────────────────────────────────
QUESTIONS_PER_RULE = 20
BATCH_SIZE = 5
SEED = 42

# ── 메뉴-복합 증강 설정 ────────────────────────────────────
# sLLM이 메뉴 노이즈를 무시하고 비메뉴 rule을 선택하도록 학습
MENU_AUG_RATIO = 0.20
MENUS = [
    "아메리카노", "아이스아메리카노", "라떼", "바닐라라떼", "딸기라떼", "에스프레소",
    "아인슈페너", "밀크티", "에이드", "수박주스", "그릭요거트",
    "크로플", "마카롱", "소금빵", "휘낭시에", "케이크", "치즈케이크", "에그타르트",
    "와플", "스콘", "베이글", "쿠키", "빙수", "푸딩", "버터바",
]
MENU_TEMPLATES = [
    "{m} 맛집인데 {q}", "{m} 맛있고 {q}", "{m} 잘하는 곳 중에 {q}",
    "{m} 먹고 싶은데 {q}", "{q} 그리고 {m}도 맛있으면 좋겠어",
]


def parse_questions(content: str) -> list[str]:
    results = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^\d+\.\s*(.+)', line)
        if match:
            results.append(match.group(1).strip())
    return results


def generate_batch(
    client: OpenAI,
    rules_batch: list[dict],
) -> dict[int, list[str]]:
    """배치로 여러 rule 질문 생성. {인덱스: [질문들]} 반환"""

    batch_prompt = ""
    for i, r in enumerate(rules_batch, 1):
        prompt = build_question_gen_prompt(r["title"], QUESTIONS_PER_RULE)
        batch_prompt += f"[Rule {i}]\n{prompt}\n\n"

    batch_prompt += "Output each rule separated by [Rule N] tags."

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": QUESTION_GEN_SYSTEM},
                {"role": "user", "content": batch_prompt},
            ],
            max_completion_tokens=3000,
            temperature=0.9,
        )
        content = response.choices[0].message.content.strip()

        results = {}
        current_idx = None
        current_lines = []

        for line in content.split("\n"):
            line = line.strip()
            tag_match = re.match(r'\[Rule (\d+)\]', line)
            if tag_match:
                if current_idx is not None and current_lines:
                    results[current_idx] = parse_questions("\n".join(current_lines))
                current_idx = int(tag_match.group(1)) - 1
                current_lines = []
            elif current_idx is not None:
                current_lines.append(line)

        if current_idx is not None and current_lines:
            results[current_idx] = parse_questions("\n".join(current_lines))

        return results

    except Exception as e:
        print(f"[ERROR] {e}")
        return {}


def augment_menu(records: list[dict], seed: int) -> list[dict]:
    """메뉴 prefix를 붙인 복합 질문 생성. gold rule(title)은 비메뉴 그대로 유지.
    sLLM이 메뉴 텍스트를 노이즈로 처리하도록 학습."""
    rng = random.Random(seed)
    n = int(len(records) * MENU_AUG_RATIO)
    sampled = rng.sample(records, n)
    augmented = []
    for d in sampled:
        m = rng.choice(MENUS)
        t = rng.choice(MENU_TEMPLATES)
        augmented.append({
            "query": t.format(m=m, q=d["query"]),
            "title": d["title"],
        })
    return augmented


def main():
    random.seed(SEED)

    with open(RULE_PATH, encoding="utf-8") as f:
        rules_data = json.load(f)
    rules = rules_data["unique_rules"]
    print(f"[INFO] 전체 rule: {len(rules)}개")
    print(f"[INFO] 목표 질문 수: {len(rules) * QUESTIONS_PER_RULE}개 (증강 전)")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_records = []

    print(f"\n[INFO] 질문 생성 시작 (rule당 {QUESTIONS_PER_RULE}개)...")
    for i in tqdm(range(0, len(rules), BATCH_SIZE)):
        batch = rules[i:i + BATCH_SIZE]
        results = generate_batch(client, batch)

        for idx, questions in results.items():
            if idx >= len(batch):
                continue
            title = batch[idx]["title"]
            for q in questions:
                all_records.append({"query": q, "title": title})

    # 메뉴-복합 증강: 생성된 질문의 MENU_AUG_RATIO만큼 추가
    augmented = augment_menu(all_records, seed=SEED)
    all_records_final = all_records + augmented

    # 셔플 후 저장
    random.shuffle(all_records_final)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for record in all_records_final:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[INFO] 완료!")
    print(f"  원본 질문: {len(all_records)}개")
    print(f"  메뉴-복합 증강: {len(augmented)}개 ({MENU_AUG_RATIO:.0%})")
    print(f"  총 저장: {len(all_records_final)}개")
    print(f"  저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
