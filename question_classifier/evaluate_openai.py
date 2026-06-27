"""
OpenAI 질문 분류기를 koelectra와 '동일한' hard_test.jsonl(300개)로 평가.

koelectra(finetune/evaluate.py)와 같은 데이터·라벨 기준으로 Accuracy / macro F1을
산출해 교체 근거를 직접 비교한다.

실행:  uv run python question_classifier/evaluate_openai.py
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify import MODEL, classify_question  # noqa: E402

DATA = Path(__file__).parent / "data_augment" / "data" / "hard_test.jsonl"

# build_dataset.py와 동일한 매핑
LABEL2ID = {"non-menu": 0, "menu-only": 1, "menu-complex": 2, "invalid": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# koelectra 기준 (README 최종 평가)
KOELECTRA = {"acc": 0.8000, "f1": 0.8084, "time": 0.0066}


def load_rows() -> list[dict]:
    rows = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _predict(row: dict) -> tuple[int, int, str, str]:
    pred_label = classify_question(row["question"])
    return LABEL2ID.get(pred_label, LABEL2ID["invalid"]), row["label"], row["question"], pred_label


def main() -> None:
    rows = load_rows()
    print(f"[INFO] hard_test {len(rows)}개 · 모델 {MODEL} · 동시 8")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_predict, rows))
    elapsed = time.time() - t0

    n = len(results)
    acc = sum(1 for p, g, _, _ in results if p == g) / n

    print(f"\n{'클래스':<13}{'P':>7}{'R':>7}{'F1':>7}   (support)")
    f1s = []
    for c in range(4):
        tp = sum(1 for p, g, _, _ in results if p == c and g == c)
        fp = sum(1 for p, g, _, _ in results if p == c and g != c)
        fn = sum(1 for p, g, _, _ in results if p != c and g == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        support = sum(1 for _, g, _, _ in results if g == c)
        print(f"{ID2LABEL[c]:<13}{prec:>7.2f}{rec:>7.2f}{f1:>7.2f}   ({support})")
    macro_f1 = sum(f1s) / len(f1s)

    print("\n" + "=" * 48)
    print(f"OpenAI ({MODEL})  Hard Test (300)")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1(macro): {macro_f1:.4f}")
    print(f"  추론     : {elapsed / n:.3f}s/건 (총 {elapsed:.1f}s, 동시 8)")
    print("-" * 48)
    print("비교 (동일 hard_test 300개, 동일 라벨 기준)")
    print(f"  {'모델':<22}{'Acc':>8}{'F1':>8}{'시간/건':>10}")
    print(f"  {'koelectra-base-v3':<22}{KOELECTRA['acc']:>8.4f}{KOELECTRA['f1']:>8.4f}{KOELECTRA['time']:>9.4f}s")
    print(f"  {f'OpenAI {MODEL}':<22}{acc:>8.4f}{macro_f1:>8.4f}{elapsed / n:>9.4f}s")
    print("=" * 48)

    # 오답 저장 (검수용)
    wrong = [
        {"question": q, "label": ID2LABEL[g], "pred": pl}
        for p, g, q, pl in results if p != g
    ]
    out = Path(__file__).parent / "data_augment" / "data" / "hard_test_openai_wrong.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for w in wrong:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    print(f"[INFO] 오답 {len(wrong)}개 → {out.name}")


if __name__ == "__main__":
    main()
