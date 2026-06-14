"""
파인튜닝 데이터 train/test 분할 스크립트
title별 stratified 9:1 분할
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "keyword_selection" / "data" / "finetune_keyword_selection.jsonl"
TRAIN_PATH = ROOT / "keyword_selection" / "data" / "train.jsonl"
TEST_PATH = ROOT / "keyword_selection" / "data" / "test.jsonl"

# ── 설정 ──────────────────────────────────────────────────
TRAIN_RATIO = 0.9
SEED = 42


def main():
    random.seed(SEED)

    print("데이터 로드 중...")
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    print(f"  전체 데이터: {len(data)}개")

    # title별로 그룹핑
    by_title = defaultdict(list)
    for d in data:
        by_title[d["answer"]].append(d)

    print(f"  title 수: {len(by_title)}개 (none 포함)")

    train_data, test_data = [], []

    for title, items in by_title.items():
        random.shuffle(items)
        train_size = max(1, int(len(items) * TRAIN_RATIO))
        train_data.extend(items[:train_size])
        test_data.extend(items[train_size:])

    random.shuffle(train_data)
    random.shuffle(test_data)

    # 저장
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for d in train_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(TEST_PATH, "w", encoding="utf-8") as f:
        for d in test_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # none 비율 확인
    train_none = sum(1 for d in train_data if d["answer"] == "none")
    test_none = sum(1 for d in test_data if d["answer"] == "none")

    print(f"\n완료!")
    print(f"  train: {len(train_data)}개 (none: {train_none}개, {train_none/len(train_data):.2%})")
    print(f"  test:  {len(test_data)}개 (none: {test_none}개, {test_none/len(test_data):.2%})")


if __name__ == "__main__":
    main()