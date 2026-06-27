"""
generic 키워드 식별.

매장 item의 키워드별 긍정 descriptor 목록을 LLM에 주고,
descriptor가 "서로 다른 의미를 가르는지(generic)" / "사실상 같은 긍정 뜻인지(단일)" 판단.

- generic  = descriptor에 따라 의미가 갈림 → cafe.db/rule에서 "키워드_descriptor"로 분리해야 함
             (예: 매장 → 넓다 / 깔끔하다 / 아늑하다)
- 단일      = descriptor가 다 비슷한 긍정 → 키워드만으로 충분
             (예: 주차장 → 넓다 / 편리하다 / 편하다,  개방감 → 개방감있다 / 넓다)

출력: data/generic_keywords.json  {"generic": [...], "single": [...]}
Usage: python classify_generic.py
"""

import json
import os
import csv
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MARKET_PATH = ROOT / "data" / "market_item.csv"
OUTPUT_PATH = ROOT / "data" / "generic_keywords.json"

SYSTEM = """You classify Korean cafe attribute keywords for a recommendation system.

For each keyword you are given its list of POSITIVE descriptors observed in cafes.
Decide whether the descriptors carry DIFFERENT meanings (so the keyword alone is ambiguous) or are essentially the SAME positive sense.

- "generic": descriptors split into clearly different meanings → the keyword alone loses information.
  e.g. 매장 → [넓다, 깔끔하다, 아늑하다]  (size vs cleanliness vs cozy — different)
       좌석 → [넓다, 많다, 공부하기 좋다]   (different aspects)
- "single": descriptors are near-synonyms / all the same positive direction → keyword alone is enough.
  e.g. 주차장 → [넓다, 편리하다, 편하다]   (all "good parking")
       개방감 → [개방감 있다, 넓다]          (same sense)

Output ONLY a JSON object: {"generic": ["키워드", ...], "single": ["키워드", ...]}
Every input keyword must appear in exactly one list."""


def main():
    with open(MARKET_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    kw_desc = defaultdict(set)
    for r in rows:
        if r["sentiment"] == "긍정" and r["대표descriptor"].strip():
            kw_desc[r["키워드"]].add(r["대표descriptor"].strip())

    # descriptor가 2종 이상인 키워드만 판단 대상 (1종 이하는 자동으로 single)
    candidates = {k: sorted(v) for k, v in kw_desc.items() if len(v) >= 2}
    auto_single = [k for k, v in kw_desc.items() if len(v) < 2]
    print(f"[INFO] 판단 대상(descriptor 2종+): {len(candidates)}개 / 자동 single: {len(auto_single)}개")

    listing = "\n".join(f"- {k}: {descs}" for k, descs in candidates.items())
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Classify these keywords:\n{listing}"},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(resp.choices[0].message.content)

    generic = sorted(set(result.get("generic", [])) & set(candidates))
    single = sorted((set(candidates) - set(generic)) | set(auto_single))

    out = {"generic": generic, "single": single}
    json.dump(out, open(OUTPUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n[generic {len(generic)}개] {generic}")
    print(f"\n[single {len(single)}개] {single[:30]} ...")
    print(f"\n저장 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
