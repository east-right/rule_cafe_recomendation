"""
임베딩 파인튜닝 학습 데이터 생성 (embedding_data.ipynb 스크립트화).

rule_metadata_merged.json의 각 rule에서:
  query = question + merged_questions (각각)
  pos   = description
  title = rule title
→ data/finetune_embd_query_pos.json

이후 embedding/finetune_bge.py가 이 파일을 train/test로 split + neg 샘플링하여 학습.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RULE_PATH = BASE_DIR / "data" / "rule_metadata_merged.json"
OUT_PATH  = BASE_DIR / "data" / "finetune_embd_query_pos.json"


def main():
    rules = json.load(open(RULE_PATH, encoding="utf-8"))["unique_rules"]

    pairs = []
    for rule in rules:
        desc  = rule["description"]
        title = rule["title"]
        if rule.get("question"):
            pairs.append({"query": rule["question"], "pos": desc, "title": title})
        for q in rule.get("merged_questions", []):
            pairs.append({"query": q, "pos": desc, "title": title})

    json.dump(pairs, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    n_rules = len(rules)
    n_q     = sum(1 for r in rules if r.get("question"))
    n_merged = sum(len(r.get("merged_questions", [])) for r in rules)
    print(f"rule {n_rules}개 → query-pos {len(pairs)}개 (대표질문 {n_q} + merged {n_merged})")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
