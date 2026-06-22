"""
sLLM 파인튜닝 질문 생성 스크립트
rule title만 기반으로 질문 생성 (구어체 + 일반체)

253개 rule x 20개 = 약 5000개

Usage: python generate_additional_data.py
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


def main():
    random.seed(SEED)

    with open(RULE_PATH, encoding="utf-8") as f:
        rules_data = json.load(f)
    rules = rules_data["unique_rules"]
    print(f"[INFO] 전체 rule: {len(rules)}개")
    print(f"[INFO] 목표 질문 수: {len(rules) * QUESTIONS_PER_RULE}개")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_generated = 0

    print(f"\n[INFO] 질문 생성 시작 (rule당 {QUESTIONS_PER_RULE}개)...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(rules), BATCH_SIZE)):
            batch = rules[i:i + BATCH_SIZE]
            results = generate_batch(client, batch)

            for idx, questions in results.items():
                if idx >= len(batch):
                    continue
                title = batch[idx]["title"]
                for q in questions:
                    record = {
                        "query": q,
                        "title": title,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_generated += 1

    print(f"\n[INFO] 완료! 총 생성: {total_generated}개")
    print(f"[INFO] 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()