import json
from collections import Counter

train = [json.loads(l) for l in open("keyword_selection/data/keyword_questions.jsonl", encoding="utf-8")]
test  = [json.loads(l) for l in open("keyword_selection/data/test.jsonl", encoding="utf-8")]

print("=== train ===")
print(f"total: {len(train)}, rules: {len(set(d['title'] for d in train))}")
vals = list(Counter(d["title"] for d in train).values())
print(f"per rule - min:{min(vals)} avg:{sum(vals)/len(vals):.1f} max:{max(vals)}")

seen = set()
print("\n--- sample (1 per rule) ---")
for d in train:
    if d["title"] not in seen:
        print(f"  [{d['title']}] {d['query']}")
        seen.add(d["title"])
    if len(seen) >= 12:
        break

print("\n=== test ===")
print(f"total: {len(test)}, rules: {len(set(d['title'] for d in test))}")
vals_t = list(Counter(d["title"] for d in test).values())
print(f"per rule - min:{min(vals_t)} avg:{sum(vals_t)/len(vals_t):.1f} max:{max(vals_t)}")

seen2 = set()
print("\n--- sample (1 per rule) ---")
for d in test:
    if d["title"] not in seen2:
        print(f"  [{d['title']}] {d['query']}")
        seen2.add(d["title"])
    if len(seen2) >= 12:
        break
