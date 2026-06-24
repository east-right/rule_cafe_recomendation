"""
BGE-M3 파인튜닝 스크립트 (하드 네거티브 버전)
train/test 분리 → 하드네거티브 채굴 → 파인튜닝

산출물:
- embedding/data/finetune_train/train.jsonl  (학습용, query/pos/neg[K])
- embedding/data/test_query_pos.json         (평가용)
- embedding/models/bge-m3-cafe/              (파인튜닝 모델)
"""

import json
import gc
import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from FlagEmbedding import BGEM3FlagModel
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
RULE_PATH       = BASE_DIR / "data" / "rule_metadata_merged.json"
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

# ── 하드 네거티브 설정 ─────────────────────────────────────────
NUM_HARD_NEG = 7          # query당 하드 네거티브 개수 (train_group_size = 1 + 이것)
HN_SKIP_TOP  = 1          # 최상위 N개는 제외(정답 유사 rule=false negative 방지)
HN_POOL      = 50         # 상위 풀에서 샘플링 (skip~skip+pool 범위)

# ── 메뉴-복합 증강 설정 (split 후 train/test 각각 적용 → 누수 방지) ─
MENU_AUG_RATIO = 0.25     # 각 split의 이 비율만큼 메뉴-복합 버전 추가
MENUS = [
    "아메리카노", "아이스아메리카노", "라떼", "바닐라라떼", "딸기라떼", "에스프레소",
    "아인슈페너", "밀크티", "에이드", "수박주스", "그릭요거트",
    "크로플", "마카롱", "소금빵", "휘낭시에", "케이크", "치즈케이크", "에그타르트",
    "와플", "스콘", "베이글", "쿠키", "빙수", "푸딩", "버터바",
]
MENU_TEMPLATES = [
    "{m} 맛집인데 {q}", "{m} 맛있고 {q}", "{m} 잘하는 곳 중에 {q}",
    "{m} 먹고 싶은데 {q}", "{q} 그리고 {m}도 맛있으면 좋겠어",
]
# ──────────────────────────────────────────────────────────────


def augment_menu(pairs, seed):
    """질문 앞에 메뉴 멘션만 붙인 버전 생성(정답 rule 그대로). 메뉴 노이즈 강건성용.
    split 이후 각 셋에 따로 적용해야 train↔test 누수가 없다."""
    rng = random.Random(seed)
    out = []
    for d in rng.sample(pairs, int(len(pairs) * MENU_AUG_RATIO)):
        m = rng.choice(MENUS)
        t = rng.choice(MENU_TEMPLATES)
        out.append({"query": t.format(m=m, q=d["query"]), "pos": d["pos"], "title": d["title"]})
    return out


def split(query_pos_path, test_path):
    """confidence stratify로 train/test 분리. test 저장하고 train_data 반환."""
    data = json.load(open(query_pos_path, encoding="utf-8"))
    rule_data = json.load(open(RULE_PATH, encoding="utf-8"))
    title_to_conf = {r["title"]: r.get("confidence", "")
                     for r in rule_data["unique_rules"]}
    labels = [title_to_conf.get(d["title"], "") for d in data]

    train_data, test_data = train_test_split(
        data, test_size=1 - TRAIN_RATIO, stratify=labels, random_state=SEED,
    )
    print(f"  전체: {len(data)}개 → train: {len(train_data)}개 / test: {len(test_data)}개")

    # 메뉴-복합 증강: split 후 각 셋에 따로 적용 (train↔test 누수 방지)
    train_data = train_data + augment_menu(train_data, SEED)
    test_data  = test_data  + augment_menu(test_data,  SEED + 1)
    print(f"  메뉴-복합 증강 후 → train: {len(train_data)}개 / test: {len(test_data)}개")

    json.dump(test_data, open(test_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  test 저장 → {test_path}")
    return train_data


def mine_and_write_train(train_data, train_path):
    """베이스 모델로 query별 하드 네거티브 채굴 후 train.jsonl 작성."""
    rules = json.load(open(RULE_PATH, encoding="utf-8"))["unique_rules"]
    pool_titles = [r["title"] for r in rules]
    pool_descs  = [r["description"] for r in rules]

    print("  🔎 하드네거티브 채굴용 베이스 모델 로드...")
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)

    def enc(texts):
        v = model.encode(texts, batch_size=64, max_length=MAX_LENGTH)["dense_vecs"]
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    print("  description / query 임베딩...")
    pool_emb  = enc(pool_descs)
    queries   = [d["query"] for d in train_data]
    q_emb     = enc(queries)

    sims = q_emb @ pool_emb.T                       # (Nq, Npool)
    ranked = np.argsort(-sims, axis=1)              # 유사도 내림차순 인덱스

    random.seed(SEED)
    with open(train_path, "w", encoding="utf-8") as f:
        for i, d in enumerate(train_data):
            gold = d["title"]
            # 같은 title(정답) 제외하고 상위 풀에서 후보 추출
            cand = [j for j in ranked[i] if pool_titles[j] != gold]
            cand = cand[HN_SKIP_TOP: HN_SKIP_TOP + HN_POOL]
            k = min(NUM_HARD_NEG, len(cand))
            negs = [pool_descs[j] for j in random.sample(cand, k)] if k else []
            f.write(json.dumps({
                "query": d["query"], "pos": [d["pos"]], "neg": negs,
            }, ensure_ascii=False) + "\n")
    print(f"  train jsonl 저장 (하드네거 {NUM_HARD_NEG}개/query) → {train_path}")

    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    print("📂 데이터 분리 중...")
    train_data = split(QUERY_POS_PATH, TEST_PATH)

    print("\n⛏️  하드 네거티브 채굴 중...")
    mine_and_write_train(train_data, TRAIN_DATA_PATH)

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
            train_group_size=1 + NUM_HARD_NEG,
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
