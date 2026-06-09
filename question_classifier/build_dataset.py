"""
전체 데이터 합치고 trainval/test 분할 스크립트
k-fold 교차검증용: trainval 90% / test 10%

Usage: python build_dataset.py
"""

import json
import random
from collections import Counter
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
VALID_QUESTIONS_PATH = ROOT / "data" / "valid_questions.json"
MENU_COMPLEX_PATH = ROOT / "question_classifier" / "data" / "menu_complex.jsonl"
INVALID_PATH = ROOT / "question_classifier" / "data" / "invalid.jsonl"
TRAINVAL_PATH = ROOT / "question_classifier" / "data" / "trainval.jsonl"
TEST_PATH = ROOT / "question_classifier" / "data" / "test.jsonl"

# ── 상수 ──────────────────────────────────────────────────
NON_MENU_SAMPLE = 600
TEST_RATIO = 0.1
SEED = 42

LABEL_MAP = {
    "non-menu": 0,
    "menu-only": 1,
    "menu-complex": 2,
    "invalid": 3,
}


def load_valid_questions() -> tuple[list[dict], list[dict]]:
    with open(VALID_QUESTIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    non_menu = [
        {"question": d["question"], "label": LABEL_MAP["non-menu"]}
        for d in data if d["type"] in ["single", "multi"]
    ]
    menu_only = [
        {"question": d["question"], "label": LABEL_MAP["menu-only"]}
        for d in data if d["type"] == "menu"
    ]
    return non_menu, menu_only


def load_jsonl(path: Path, label_key: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            records.append({
                "question": d["question"],
                "label": LABEL_MAP[label_key],
            })
    return records


def split_test(data: list[dict]) -> tuple[list[dict], list[dict]]:
    indices = list(range(len(data)))
    random.shuffle(indices)
    test_size = int(len(data) * TEST_RATIO)
    test_idx = indices[:test_size]
    trainval_idx = indices[test_size:]
    return (
        [data[i] for i in trainval_idx],
        [data[i] for i in test_idx],
    )


def main():
    random.seed(SEED)

    print("[INFO] 데이터 로드 중...")
    non_menu, menu_only = load_valid_questions()
    menu_complex = load_jsonl(MENU_COMPLEX_PATH, "menu-complex")
    invalid = load_jsonl(INVALID_PATH, "invalid")

    non_menu = random.sample(non_menu, NON_MENU_SAMPLE)

    print(f"  non-menu:     {len(non_menu)}개")
    print(f"  menu-only:    {len(menu_only)}개")
    print(f"  menu-complex: {len(menu_complex)}개")
    print(f"  invalid:      {len(invalid)}개")
    print(f"  전체:         {len(non_menu) + len(menu_only) + len(menu_complex) + len(invalid)}개")

    # ── 클래스별 stratified trainval/test 분할 ────────────
    trainval_data, test_data = [], []

    for class_data in [non_menu, menu_only, menu_complex, invalid]:
        trainval, test = split_test(class_data)
        trainval_data.extend(trainval)
        test_data.extend(test)

    random.shuffle(trainval_data)
    random.shuffle(test_data)

    # ── 저장 ──────────────────────────────────────────────
    for path, split in [(TRAINVAL_PATH, trainval_data), (TEST_PATH, test_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for d in split:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    label_names = {v: k for k, v in LABEL_MAP.items()}

    print(f"\n[INFO] 완료!")
    print(f"  trainval: {len(trainval_data)}개 (k-fold 학습/검증용)")
    print(f"  test:     {len(test_data)}개 (최종 평가용)")

    for split_name, split in [("trainval", trainval_data), ("test", test_data)]:
        dist = Counter(d["label"] for d in split)
        print(f"\n  {split_name} 분포:")
        for label, count in sorted(dist.items()):
            print(f"    {label_names[label]}: {count}개")


if __name__ == "__main__":
    main()