"""
골든셋 생성 스크립트 (수작업 라벨링용)
data/golden_set.csv 생성 후 canonical 컬럼을 직접 채워주세요.
null 제외 항목은 canonical에 'null' 기입.
"""

import json, random, csv, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

data = json.load(open(DATA / "ner_result_v2.json", encoding="utf-8"))

ec = collections.defaultdict(collections.Counter)
dc = collections.defaultdict(collections.Counter)
for entities in data.values():
    for e in entities:
        cat = e["category"]
        ec[cat][e["entity"]] += 1
        if e.get("descriptor") and e["type"] == "EVALUATIVE":
            dc[cat][e["descriptor"]] += 1

# 카테고리별 샘플 수 (상위/중간/하위 균등 샘플링)
SAMPLE_CONFIG = {
    ("MENU",       "entity"):     35,
    ("FACILITY",   "entity"):     25,
    ("ATMOSPHERE", "entity"):     20,
    ("TARGET",     "entity"):     20,
    ("MENU",       "descriptor"): 30,
    ("FACILITY",   "descriptor"): 25,
    ("ATMOSPHERE", "descriptor"): 20,
}

def stratified_sample(items: list, n: int) -> list:
    if len(items) <= n:
        return [k for k, _ in items]
    t = n // 3
    top    = [k for k, _ in items[:t]]
    mid_pool = [k for k, _ in items[len(items)//3 : 2*len(items)//3]]
    bot_pool = [k for k, _ in items[-len(items)//3 :]]
    mid    = random.sample(mid_pool, min(t, len(mid_pool)))
    bot    = random.sample(bot_pool, min(n - t*2, len(bot_pool)))
    return top + mid + bot

random.seed(42)
rows = []
for (cat, axis), n in SAMPLE_CONFIG.items():
    counter = ec[cat] if axis == "entity" else dc[cat]
    sampled = stratified_sample(counter.most_common(), n)
    for s in sampled:
        rows.append({
            "category":  cat,
            "axis":      axis,
            "original":  s,
            "freq":      counter[s],
            "canonical": "",
            "note":      "",
        })

random.shuffle(rows)

out = DATA / "golden_set.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["category","axis","original","freq","canonical","note"])
    w.writeheader()
    w.writerows(rows)

print(f"골든셋 생성 완료: {len(rows)}개 항목 → {out}")
print("canonical 컬럼을 채우고 normalize.py 실행 후 검증하세요.")
