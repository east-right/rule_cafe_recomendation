"""
전체 데이터 합치고 train/val/test 분할 스크립트
8:1:1 stratified split

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
TRAIN_PATH = ROOT / "question_classifier" / "data" / "train.jsonl"
VAL_PATH = ROOT / "question_classifier" / "data" / "val.jsonl"
TEST_PATH = ROOT / "question_classifier" / "data" / "test.jsonl"

# ── 상수 ──────────────────────────────────────────────────
NON_MENU_SAMPLE = 600
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
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


def stratified_split(data: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    indices = list(range(len(data)))
    random.shuffle(indices)
    train_size = int(len(data) * TRAIN_RATIO)
    val_size = int(len(data) * VAL_RATIO)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    return (
        [data[i] for i in train_idx],
        [data[i] for i in val_idx],
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

    # ── 클래스별 stratified 8:1:1 분할 ───────────────────
    train_data, val_data, test_data = [], [], []

    for class_data in [non_menu, menu_only, menu_complex, invalid]:
        tr, va, te = stratified_split(class_data)
        train_data.extend(tr)
        val_data.extend(va)
        test_data.extend(te)

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # ── 저장 ──────────────────────────────────────────────
    for path, split in [(TRAIN_PATH, train_data), (VAL_PATH, val_data), (TEST_PATH, test_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for d in split:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    label_names = {v: k for k, v in LABEL_MAP.items()}

    print(f"\n[INFO] 완료!")
    print(f"  train: {len(train_data)}개")
    print(f"  val:   {len(val_data)}개")
    print(f"  test:  {len(test_data)}개")

    for split_name, split in [("train", train_data), ("val", val_data), ("test", test_data)]:
        dist = Counter(d["label"] for d in split)
        print(f"\n  {split_name} 분포:")
        for label, count in sorted(dist.items()):
            print(f"    {label_names[label]}: {count}개")


if __name__ == "__main__":
    main()