"""
파인튜닝 데이터 train/test 분할 스크립트
9:1 비율로 분할
"""

import json
import random
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

    # none/정답 비율 유지하면서 스플릿 (stratified)
    match_data = [d for d in data if d["answer"] != "none"]
    none_data = [d for d in data if d["answer"] == "none"]

    print(f"  정답 있는 데이터: {len(match_data)}개")
    print(f"  none 데이터: {len(none_data)}개")

    random.shuffle(match_data)
    random.shuffle(none_data)

    # 각각 9:1 분할
    match_train_size = int(len(match_data) * TRAIN_RATIO)
    none_train_size = int(len(none_data) * TRAIN_RATIO)

    train_data = match_data[:match_train_size] + none_data[:none_train_size]
    test_data = match_data[match_train_size:] + none_data[none_train_size:]

    # 섞기
    random.shuffle(train_data)
    random.shuffle(test_data)

    # 저장
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for d in train_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(TEST_PATH, "w", encoding="utf-8") as f:
        for d in test_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n완료!")
    print(f"  train: {len(train_data)}개")
    print(f"  test:  {len(test_data)}개")
    print(f"  train none 비율: {sum(1 for d in train_data if d['answer'] == 'none') / len(train_data):.2%}")
    print(f"  test  none 비율: {sum(1 for d in test_data if d['answer'] == 'none') / len(test_data):.2%}")


if __name__ == "__main__":
    main()
