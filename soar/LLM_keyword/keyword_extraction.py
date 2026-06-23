import asyncio
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from prompt import RULE_KEYWORD_EXTRACTION_SYSTEM, RULE_KEYWORD_EXTRACTION_USER

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_PATH = Path("../../data/rule_metadata_merged.json")
OUTPUT_PATH = Path("../../data/soar_rule_keywords.json")
GENERIC_PATH = Path("../../data/generic_keywords.json")
MARKET_PATH = Path("../../data/market_item.csv")

# generic 키워드 → 매장 item의 긍정 descriptor 목록
GENERIC = set(json.load(open(GENERIC_PATH, encoding="utf-8"))["generic"])
GENERIC_DESCS: dict[str, list[str]] = defaultdict(list)
with open(MARKET_PATH, encoding="utf-8-sig") as _f:
    _seen = set()
    for _r in csv.DictReader(_f):
        _kw, _d = _r["키워드"], _r["대표descriptor"].strip()
        if _kw in GENERIC and _d and _r["sentiment"] == "긍정" and (_kw, _d) not in _seen:
            _seen.add((_kw, _d))
            GENERIC_DESCS[_kw].append(f"{_kw}_{_d}")


def expand_keywords(keywords: list[str]) -> list[str]:
    """generic 키워드는 '키워드_descriptor' 후보들로 확장(cafe.db와 동일 형식),
    나머지는 그대로. → operator/tiebreak가 매장 item과 매칭되는 형식으로 선택되게 함."""
    pool: list[str] = []
    for kw in keywords:
        if kw in GENERIC and GENERIC_DESCS.get(kw):
            pool.extend(GENERIC_DESCS[kw])
        else:
            pool.append(kw)
    # 순서 보존 중복 제거
    seen, out = set(), []
    for k in pool:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


async def extract_keywords(rule: dict) -> dict:
    user_prompt = RULE_KEYWORD_EXTRACTION_USER.format(
        title=rule["title"],
        description=rule["description"],
        keywords=expand_keywords(rule["keywords"]),
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


def sanitize_result(result: dict, rule: dict) -> dict:
    """풀 이탈 키워드만 제거하고 나머지는 살린다. operator가 다 이탈하면
    확장 풀 앞부분으로 fallback (통째로 버려 rule이 누락되는 것 방지)."""
    pool = expand_keywords(rule["keywords"])
    keyword_pool = set(pool)

    ops = [kw for kw in result.get("operator_keywords", []) if kw in keyword_pool]
    tbs = [kw for kw in result.get("tiebreak_keywords", []) if kw in keyword_pool]

    dropped = [kw for kw in result.get("operator_keywords", []) + result.get("tiebreak_keywords", [])
               if kw not in keyword_pool]
    if dropped:
        print(f"[{rule['title']}] 풀 이탈 키워드 제거: {dropped}")

    if not ops:
        ops = pool[:5]
        print(f"[{rule['title']}] operator 전부 이탈 → 확장 풀 앞 5개로 fallback")

    return {
        "title": rule["title"],
        "operator_keywords": ops,
        "tiebreak_keywords": tbs,
    }


def make_single_keyword_rule(rule: dict) -> dict:
    """키워드 1개짜리 rule은 LLM 없이 바로 생성 (generic이면 descriptor 확장 적용)"""
    return {
        "title": rule["title"],
        "operator_keywords": expand_keywords(rule["keywords"]),
        "tiebreak_keywords": []
    }


async def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        rules = data["unique_rules"]

    # 중복 처리
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

    # 키워드 1개면 LLM 스킵
    single_rules = [r for r in rules if len(r["keywords"]) == 1]
    llm_rules = [r for r in rules if len(r["keywords"]) > 1]

    results = []
    for rule in single_rules:
        results.append(make_single_keyword_rule(rule))

    print(f"단일 키워드 rule (LLM 스킵): {len(single_rules)}개")
    print(f"총 {len(llm_rules)}개 rule LLM 처리 시작")

    tasks = [extract_keywords(rule) for rule in llm_rules]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    failed = []
    for rule, response in zip(llm_rules, responses):
        if isinstance(response, Exception):
            print(f"[{rule['title']}] 오류: {response}")
            failed.append(rule["title"])
        else:
            results.append(sanitize_result(response, rule))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(results)}개 성공, {len(failed)}개 실패")
    if failed:
        print(f"실패 목록: {failed}")


if __name__ == "__main__":
    asyncio.run(main())