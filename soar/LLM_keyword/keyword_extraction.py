# soar/LLM_keyword/keyword_extraction.py

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from prompt import RULE_KEYWORD_EXTRACTION_SYSTEM, RULE_KEYWORD_EXTRACTION_USER

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_PATH = Path("../../data/rule_metadata_merged.json")
OUTPUT_PATH = Path("../../data/soar_rule_keywords.json")


async def extract_keywords(rule: dict) -> dict:
    user_prompt = RULE_KEYWORD_EXTRACTION_USER.format(
        title=rule["title"],
        description=rule["description"],
        keywords=rule["keywords"],
    )

    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": RULE_KEYWORD_EXTRACTION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result


def validate_result(result: dict, rule: dict) -> bool:
    keyword_pool = set(rule["keywords"])

    for kw in result.get("operator_keywords", []):
        if kw not in keyword_pool:
            print(f"[{rule['title']}] operator_keyword 풀 이탈: {kw}")
            return False

    for kw in result.get("tiebreak_keywords", []):
        if kw not in keyword_pool:
            print(f"[{rule['title']}] tiebreak_keyword 풀 이탈: {kw}")
            return False

    if not (4 <= len(result.get("operator_keywords", [])) <= 5):
        print(f"[{rule['title']}] operator_keywords 개수 오류: {len(result.get('operator_keywords', []))}")
        return False

    if not (4 <= len(result.get("tiebreak_keywords", [])) <= 5):
        print(f"[{rule['title']}] tiebreak_keywords 개수 오류: {len(result.get('tiebreak_keywords', []))}")
        return False

    return True


async def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        rules = data["unique_rules"]

    # title 기준 중복 처리 - keywords 합집합
    merged = {}
    for rule in rules:
        title = rule["title"]
        if title not in merged:
            merged[title] = rule.copy()
            merged[title]["keywords"] = set(rule["keywords"])
        else:
            merged[title]["keywords"] |= set(rule["keywords"])

    rules = []
    for rule in merged.values():
        rule["keywords"] = list(rule["keywords"])
        rules.append(rule)

    print(f"중복 처리 후: {len(rules)}개 rule")
    print(f"총 {len(rules)}개 rule 처리 시작")
    

    print(f"총 {len(rules)}개 rule 처리 시작")

    tasks = [extract_keywords(rule) for rule in rules]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    failed = []

    for rule, response in zip(rules, responses):
        if isinstance(response, Exception):
            print(f"[{rule['title']}] 오류: {response}")
            failed.append(rule["title"])
        elif validate_result(response, rule):
            results.append(response)
        else:
            failed.append(rule["title"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(results)}개 성공, {len(failed)}개 실패")
    if failed:
        print(f"실패 목록: {failed}")


if __name__ == "__main__":
    asyncio.run(main())