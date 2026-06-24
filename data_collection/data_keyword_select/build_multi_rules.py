"""
독립 조합 multi rule 생성 (v3)

기존 multi(빈티지감성, 쾌적향기 등 동의어 조합)를 전부 버리고,
single rule 97개만 재료로 "서로 독립적인" 두 속성을 AND로 묶는 복합 rule을 새로 생성한다.

설계:
- multi의 가치 = 두 조건을 동시에 만족해야만 하는 경우 (한쪽 추천으로 다른 쪽이 커버 안 됨)
- 동의어/근접 조합(여유+조용함 등)은 single로 충분하니 제외
- 임베딩 유사도는 보조(동의어 가이드 + 사후 컷), 실제 조합 판단은 LLM이 수행

입력 : data/rule_metadata_merged.json 의 single(=confidence!='multi') 97개
출력 : data/rule_metadata_merged.json (single 유지 + 새 multi)
백업 : data/rule_metadata_merged_premulti.json

Usage: python build_multi_rules.py
"""

import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI

from prompt import MULTI_RULE_GENERATION_SYSTEM_PROMPT, build_multi_generation_prompt

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

RULE_PATH = ROOT / "data" / "rule_metadata_merged.json"
BACKUP_PATH = ROOT / "data" / "rule_metadata_merged_premulti.json"
EMB_MODEL = "east-right/bge-m3-cafe-finetuned"

# ── 설정 ──────────────────────────────────────────────────
TARGET_MULTI = 80          # 목표 multi 개수
BATCH_N = 20               # LLM 1회 호출당 생성 수
SYN_GUIDE_THRESH = 0.82    # 프롬프트에 동의어 가이드로 줄 유사도 컷
SYN_REJECT_THRESH = 0.85   # 생성 결과 사후 동의어 제거 컷
MODEL = "gpt-4.1"


def load_singles() -> list[dict]:
    data = json.load(open(RULE_PATH, encoding="utf-8"))
    return [r for r in data["unique_rules"] if r.get("confidence") != "multi"]


def embed(descs: list[str], model) -> np.ndarray:
    v = model.encode(descs, batch_size=16, max_length=512)["dense_vecs"]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def synonym_pairs(titles, sim, thresh) -> list[tuple[str, str]]:
    n = len(titles)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= thresh:
                pairs.append((titles[i], titles[j]))
    return pairs


def parse_json_array(content: str) -> list[dict]:
    content = content.strip()
    content = re.sub(r"^```(json)?|```$", "", content, flags=re.MULTILINE).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 배열만 추출 시도
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return []


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    singles = load_singles()
    titles = [r["title"] for r in singles]
    title_set = set(titles)
    print(f"[INFO] single: {len(singles)}개")

    print("[INFO] 임베딩 로드 + 유사도 계산...")
    model = BGEM3FlagModel(EMB_MODEL, use_fp16=True)
    desc_emb = embed([r["description"] for r in singles], model)
    sim = desc_emb @ desc_emb.T
    idx = {t: i for i, t in enumerate(titles)}

    syn_for_prompt = synonym_pairs(titles, sim, SYN_GUIDE_THRESH)
    banned_str = "\n".join(f"- {a} + {b}" for a, b in syn_for_prompt[:60])
    print(f"[INFO] 동의어 가이드 쌍: {len(syn_for_prompt)}개")

    single_lines = "\n".join(
        f"- {r['title']}: {r['description'][:50]}" for r in singles
    )

    # ── LLM 배치 생성 ─────────────────────────────────────
    collected: dict[frozenset, dict] = {}
    rounds = 0
    while len(collected) < TARGET_MULTI and rounds < 8:
        rounds += 1
        avoid = ", ".join(sorted(c["title"] for c in collected.values()))[:1500]
        user = build_multi_generation_prompt(single_lines, BATCH_N, banned_str, avoid)
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": MULTI_RULE_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.8,
                max_completion_tokens=3000,
            )
            items = parse_json_array(resp.choices[0].message.content)
        except Exception as e:
            print(f"  [WARN] round {rounds}: {e}")
            continue

        added = 0
        for it in items:
            m1, m2 = it.get("member1", ""), it.get("member2", "")
            # 멤버가 실제 single 목록에 있어야 함
            if m1 not in title_set or m2 not in title_set or m1 == m2:
                continue
            key = frozenset([m1, m2])
            if key in collected:
                continue
            # 사후 동의어 컷
            if sim[idx[m1], idx[m2]] >= SYN_REJECT_THRESH:
                continue
            it["members"] = [m1, m2]
            it["_sim"] = float(sim[idx[m1], idx[m2]])
            collected[key] = it
            added += 1
        print(f"  round {rounds}: +{added} (누적 {len(collected)})")

    multis = list(collected.values())[:TARGET_MULTI]
    print(f"\n[INFO] 최종 multi: {len(multis)}개")
    print(f"  멤버쌍 유사도 평균 {np.mean([m['_sim'] for m in multis]):.3f} "
          f"(min {min(m['_sim'] for m in multis):.3f} / max {max(m['_sim'] for m in multis):.3f})")

    # ── 새 rule_metadata 빌드 ─────────────────────────────
    new_multis = []
    for m in multis:
        new_multis.append({
            "title": m["title"],
            "description": m["description"],
            "keywords": [],
            "negative_keywords": [],
            "question": "",
            "merged_questions": [],
            "members": m["members"],
            "source_count": 0,
            "confidence": "multi",
            "reason": "independent AND 조합 (single 두 속성 결합)",
        })

    # 백업
    orig = json.load(open(RULE_PATH, encoding="utf-8"))
    json.dump(orig, open(BACKUP_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[INFO] 백업 → {BACKUP_PATH}")

    out = {"unique_rules": singles + new_multis}
    json.dump(out, open(RULE_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[INFO] 저장 → {RULE_PATH}  (single {len(singles)} + multi {len(new_multis)} = {len(out['unique_rules'])})")


if __name__ == "__main__":
    main()
