"""153개 질문에서 평가용 50개를 층화 샘플 → questions_eval.json.

- src(테마) 비율을 유지해 다양성 보존(공부·만남·메뉴·분위기·트레이드오프·none 등 골고루).
- 원래 qid 유지(추적용). questions.json(153)은 superset으로 보존.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import batch_runner as br  # stdout utf-8

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "questions.json"
OUT = HERE / "data" / "questions_eval.json"
TARGET = 50

qs = json.loads(SRC.read_text(encoding="utf-8"))
by_src = defaultdict(list)
for q in qs:
    by_src[q["src"]].append(q)

random.seed(0)
for items in by_src.values():
    random.shuffle(items)

# 비례 배분 후 합계를 정확히 TARGET으로 보정
alloc = {s: round(TARGET * len(v) / len(qs)) for s, v in by_src.items()}
order = sorted(by_src, key=lambda s: len(by_src[s]), reverse=True)
while sum(alloc.values()) > TARGET:
    for s in order:
        if alloc[s] > 1:
            alloc[s] -= 1
            if sum(alloc.values()) == TARGET:
                break
while sum(alloc.values()) < TARGET:
    for s in order:
        if alloc[s] < len(by_src[s]):
            alloc[s] += 1
            if sum(alloc.values()) == TARGET:
                break

selected = []
for s in by_src:
    selected += by_src[s][:alloc[s]]
selected.sort(key=lambda q: q["qid"])

OUT.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"선택: {len(selected)}개  (원본 {len(qs)}개)")
print("\n테마별:", {s: alloc[s] for s in by_src})
print(f"\n저장: {OUT}\n")
for q in selected:
    print(f'{q["qid"]}  [{q["src"]:>12}]  {q["question"]}')
