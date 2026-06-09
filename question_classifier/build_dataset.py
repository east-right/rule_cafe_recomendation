"""
전체 데이터 합치고 train/test 분할 스크립트

클래스별 데이터:
- non-menu: valid_questions.json (single+multi) → 600개 샘플링
- menu-only: valid_questions.json (menu) → 559개 전부
- menu-complex: menu_complex.jsonl → 600개
- invalid: invalid.jsonl → 600개

Usage: python build_dataset.py
"""

import json
import random
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
VALID_QUESTIONS_PATH = ROOT / "data" / "valid_questions.json"
MENU_COMPLEX_PATH = ROOT / "question_classifier" / "data" / "menu_complex.jsonl"
INVALID_PATH = ROOT / "question_classifier" / "data" / "invalid.jsonl"
TRAIN_PATH = ROOT / "question_classifier" / "data" / "train.jsonl"
TEST_PATH = ROOT / "question_classifier" / "data" / "test.jsonl"

# ── 상수 ──────────────────────────────────────────────────
NON_MENU_SAMPLE = 600
TRAIN_RATIO = 0.9
SEED = 42

# ── 라벨 매핑 ─────────────────────────────────────────────
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


def split_data(data: list[dict], ratio: float) -> tuple[list[dict], list[dict]]:
    random.shuffle(data)
    train_size = int(len(data) * ratio)
    return data[:train_size], data[train_size:]


def main():
    random.seed(SEED)

    # ── 데이터 로드 ───────────────────────────────────────
    print("[INFO] 데이터 로드 중...")
    non_menu, menu_only = load_valid_questions()
    menu_complex = load_jsonl(MENU_COMPLEX_PATH, "menu-complex")
    invalid = load_jsonl(INVALID_PATH, "invalid")

    # non-menu 샘플링
    non_menu = random.sample(non_menu, NON_MENU_SAMPLE)

    print(f"  non-menu:     {len(non_menu)}개")
    print(f"  menu-only:    {len(menu_only)}개")
    print(f"  menu-complex: {len(menu_complex)}개")
    print(f"  invalid:      {len(invalid)}개")
    print(f"  전체:         {len(non_menu) + len(menu_only) + len(menu_complex) + len(invalid)}개")

    # ── 클래스별 train/test 분할 (stratified) ─────────────
    train_data, test_data = [], []

    for class_data in [non_menu, menu_only, menu_complex, invalid]:
        train, test = split_data(class_data, TRAIN_RATIO)
        train_data.extend(train)
        test_data.extend(test)

    # 최종 셔플
    random.shuffle(train_data)
    random.shuffle(test_data)

    # ── 저장 ──────────────────────────────────────────────
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for d in train_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(TEST_PATH, "w", encoding="utf-8") as f:
        for d in test_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n[INFO] 완료!")
    print(f"  train: {len(train_data)}개")
    print(f"  test:  {len(test_data)}개")

    # 클래스 분포 확인
    from collections import Counter
    label_names = {v: k for k, v in LABEL_MAP.items()}
    train_dist = Counter(d["label"] for d in train_data)
    test_dist = Counter(d["label"] for d in test_data)

    print(f"\n  train 분포:")
    for label, count in sorted(train_dist.items()):
        print(f"    {label_names[label]}: {count}개")

    print(f"\n  test 분포:")
    for label, count in sorted(test_dist.items()):
        print(f"    {label_names[label]}: {count}개")


if __name__ == "__main__":
    main()
