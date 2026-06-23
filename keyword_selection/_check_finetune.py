import json
from collections import Counter

for mode, path in [
    ("train", "keyword_selection/data/finetune_keyword_selection.jsonl"),
    ("test",  "keyword_selection/data/test_with_candidates.jsonl"),
]:
    data = [json.loads(l) for l in open(path, encoding="utf-8")]
    answers = [d["answer"] for d in data]
    none_cnt = answers.count("none")
    match_cnt = len(answers) - none_cnt
    total = len(data)

    print(f"=== {mode} ===")
    print(f"total: {total}")
    print(f"  match: {match_cnt} ({match_cnt/total:.1%})")
    print(f"  none : {none_cnt} ({none_cnt/total:.1%})")
    print(f"  candidates: {min(len(d['candidates']) for d in data)}~{max(len(d['candidates']) for d in data)}개")
    print()
