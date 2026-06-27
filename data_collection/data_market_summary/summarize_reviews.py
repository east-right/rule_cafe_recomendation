"""
매장 리뷰 요약 v2 — 3소스(시그니처 메뉴 + 긍정/부정 원본 리뷰)를 LLM에 던져 요약 → cafes 적재

소스:
  1. 시그니처 메뉴 : menu_item.csv 매장별 맛집score 상위 3개
  2. 긍정 원본 리뷰 : ner_result_v3.json sentiment=긍정, 키워드(entity) 빈도순 다양성 10개
  3. 부정 원본 리뷰 : 〃 sentiment=부정, 다양성 최대 5개 (없으면 생략)

동기 호출(asyncio + Semaphore)로 처리. 배치 아님.
Usage: python summarize_reviews.py
"""

import asyncio
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from prompt import REVIEW_SUMMARY_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MENU_CSV = ROOT / "data" / "menu_item.csv"
MARKET_CSV = ROOT / "data" / "market_item.csv"
NER_PATH = ROOT / "data" / "ner_result_v3.json"
DB_PATH = ROOT / "data" / "cafe.db"

MODEL = "gpt-4.1"
N_SIGNATURE = 3
N_POS = 10
N_NEG = 5
SEM = asyncio.Semaphore(3)  # gpt-4.1 TPM 30k 제한 → 동시성 낮춤


def load_pid2name() -> dict[str, str]:
    """place_id → 사업장명 (market_item + menu_item 합집합)."""
    mapping: dict[str, str] = {}
    for path in (MARKET_CSV, MENU_CSV):
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("place_id") and r.get("사업장명"):
                    mapping[r["place_id"]] = r["사업장명"]
    return mapping


def load_signatures() -> dict[str, list[str]]:
    """매장명 → 맛집score 상위 N_SIGNATURE 메뉴명."""
    by_cafe: dict[str, list[tuple[float, str]]] = defaultdict(list)
    with open(MENU_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            name, menu = r["사업장명"], r["specific"]
            if not name or not menu:
                continue
            try:
                score = float(r["맛집score"]) if r["맛집score"] else 0.0
            except ValueError:
                score = 0.0
            by_cafe[name].append((score, menu))
    result = {}
    for name, items in by_cafe.items():
        items.sort(key=lambda x: -x[0])
        seen, top = set(), []
        for _, menu in items:
            if menu in seen:
                continue
            seen.add(menu)
            top.append(menu)
            if len(top) >= N_SIGNATURE:
                break
        result[name] = top
    return result


def pick_reviews(items: list[dict], sentiment: str, limit: int) -> list[str]:
    """키워드(entity) 빈도순으로 다양성 있게 원본 리뷰 선정 (entity당 1개, 중복 제거)."""
    by_entity: dict[str, list[str]] = defaultdict(list)
    for it in items:
        if it.get("sentiment") == sentiment and it.get("source_review"):
            by_entity[it["entity"]].append(it["source_review"])
    # entity 등장 빈도 내림차순 (자주 언급된 측면 우선)
    order = sorted(by_entity, key=lambda e: -len(by_entity[e]))
    seen, out = set(), []
    for e in order:
        rv = by_entity[e][0].strip()
        if rv in seen:
            continue
        seen.add(rv)
        out.append(rv)
        if len(out) >= limit:
            break
    return out


def build_inputs() -> dict[str, dict]:
    pid2name = load_pid2name()
    signatures = load_signatures()
    ner = json.load(open(NER_PATH, encoding="utf-8"))

    inputs: dict[str, dict] = {}
    for pid, items in ner.items():
        name = pid2name.get(pid)
        if not name:
            continue
        inputs[name] = {
            "signature": signatures.get(name, []),
            "pos": pick_reviews(items, "긍정", N_POS),
            "neg": pick_reviews(items, "부정", N_NEG),
        }
    return inputs


def build_user_content(name: str, d: dict) -> str:
    sig = ", ".join(d["signature"]) if d["signature"] else "(정보 없음)"
    pos = "\n".join(f"- {r}" for r in d["pos"]) or "(없음)"
    neg = "\n".join(f"- {r}" for r in d["neg"])
    parts = [f"매장명: {name}", f"\n[시그니처 메뉴]\n{sig}", f"\n[긍정 리뷰]\n{pos}"]
    if neg:
        parts.append(f"\n[부정 리뷰]\n{neg}")
    return "\n".join(parts)


async def summarize(client: AsyncOpenAI, name: str, d: dict) -> tuple[str, str]:
    content = build_user_content(name, d)
    async with SEM:
        for attempt in range(6):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": REVIEW_SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=400,
                    temperature=0.3,
                )
                return name, resp.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e) and attempt < 5:
                    await asyncio.sleep(3.0 * (attempt + 1))
                    continue
                print(f"  [WARN] {name}: {e}")
                return name, ""


def update_db(results: dict[str, str]) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executemany(
            "UPDATE cafes SET review_summary = ? WHERE name = ?",
            [(s, n) for n, s in results.items() if s],
        )
        conn.commit()
        cnt = conn.execute(
            "SELECT COUNT(*) FROM cafes WHERE review_summary IS NOT NULL AND review_summary != ''"
        ).fetchone()[0]
        print(f"[INFO] DB 업데이트 완료: review_summary 보유 {cnt}개")
    finally:
        conn.close()


async def main() -> None:
    client = AsyncOpenAI()
    print("[INFO] 3소스 입력 빌드 중...")
    inputs = build_inputs()
    n_neg = sum(1 for d in inputs.values() if d["neg"])
    print(f"  매장 {len(inputs)}개 (부정 리뷰 있는 매장 {n_neg}개)")

    print(f"[INFO] 요약 생성 중 ({MODEL}, 동시 {SEM._value})...")
    tasks = [summarize(client, name, d) for name, d in inputs.items()]
    done = await asyncio.gather(*tasks)
    results = {name: summary for name, summary in done}

    ok = sum(1 for s in results.values() if s)
    print(f"[INFO] 요약 완료: 성공 {ok} / 전체 {len(results)}")
    update_db(results)
    print("[INFO] 전체 완료")


if __name__ == "__main__":
    asyncio.run(main())
