"""
BGE-M3 파인튜닝 스크립트
train/test 분리 후 파인튜닝

산출물:
- embedding/data/finetune_train/train.jsonl  (학습용)
- embedding/data/test_query_pos.json         (평가용)
- embedding/models/bge-m3-cafe/              (파인튜닝 모델)
"""

import json
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from FlagEmbedding.finetune.embedder.encoder_only.base import (
    EncoderOnlyEmbedderRunner,
    EncoderOnlyEmbedderDataArguments,
    EncoderOnlyEmbedderModelArguments,
    EncoderOnlyEmbedderTrainingArguments,
)

load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parents[1]
EMB_DIR         = Path(__file__).resolve().parent
EMB_DATA_DIR    = EMB_DIR / "data"
EMB_DATA_DIR.mkdir(exist_ok=True)

QUERY_POS_PATH  = BASE_DIR / "data" / "finetune_embd_query_pos.json"
TRAIN_DIR       = EMB_DATA_DIR / "finetune_train"
TRAIN_DIR.mkdir(exist_ok=True)
TRAIN_DATA_PATH = TRAIN_DIR / "train.jsonl"
TEST_PATH       = EMB_DATA_DIR / "test_query_pos.json"
OUTPUT_DIR      = EMB_DIR / "models" / "bge-m3-cafe"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 파인튜닝 설정 ──────────────────────────────────────────────
MODEL_NAME   = "BAAI/bge-m3"
EPOCHS       = 5
BATCH_SIZE   = 16
LR           = 1e-5
MAX_LENGTH   = 512
TEMPERATURE  = 0.02
TRAIN_RATIO  = 0.8
SEED         = 42
# ──────────────────────────────────────────────────────────────


from sklearn.model_selection import train_test_split

def split_and_convert(query_pos_path, train_path, test_path):
    with open(query_pos_path, encoding="utf-8") as f:
        data = json.load(f)

    rule_path = Path(__file__).resolve().parents[1] / "data" / "rule_metadata_merged.json"
    with open(rule_path, encoding="utf-8") as f:
        rule_data = json.load(f)
    title_to_conf = {r["title"]: r.get("confidence", "")
                     for r in rule_data["unique_rules"]}

    labels = [title_to_conf.get(d["title"], "") for d in data]

    train_data, test_data = train_test_split(
        data,
        test_size=1 - TRAIN_RATIO,
        stratify=labels,
        random_state=SEED,
    )

    print(f"  전체: {len(data)}개 → train: {len(train_data)}개 / test: {len(test_data)}개")

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"  test 저장 → {test_path}")

    descriptions = [d["pos"] for d in train_data]
    random.seed(SEED)
    with open(train_path, "w", encoding="utf-8") as f:
        for i, d in enumerate(train_data):
            neg_idx = random.choice([j for j in range(len(descriptions)) if j != i])
            row = {
                "query": d["query"],
                "pos":   [d["pos"]],
                "neg":   [descriptions[neg_idx]],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  train jsonl 저장 → {train_path}")


def main():
    print("📂 데이터 분리 및 변환 중...")
    split_and_convert(QUERY_POS_PATH, TRAIN_DATA_PATH, TEST_PATH)

    print("\n🚀 파인튜닝 시작...")
    runner = EncoderOnlyEmbedderRunner(
        model_args=EncoderOnlyEmbedderModelArguments(
            model_name_or_path=MODEL_NAME,
        ),
        data_args=EncoderOnlyEmbedderDataArguments(
            train_data=[str(TRAIN_DIR.resolve())],
            query_max_len=MAX_LENGTH,
            passage_max_len=MAX_LENGTH,
            query_instruction_for_retrieval="",
            passage_instruction_for_retrieval="",
            train_group_size=2,
        ),
        training_args=EncoderOnlyEmbedderTrainingArguments(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LR,
            temperature=TEMPERATURE,
            fp16=True,
            logging_steps=10,
            save_strategy="no",
            negatives_cross_device=False,
        ),
    )
    runner.run()
    print(f"\n✅ 파인튜닝 완료! → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
