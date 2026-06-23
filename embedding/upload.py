import os
from huggingface_hub import HfApi
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
print(repr(os.environ.get("HUGGINGFACE_TOKEN_WRITE")))
api = HfApi(token=os.environ.get("HUGGINGFACE_TOKEN_WRITE"))

EMB_DIR         = Path(__file__).resolve().parent
MODEL_DIR       = EMB_DIR / "models" / "bge-m3-cafe"
EMB_DATA_DIR    = EMB_DIR / "data"

MODEL_REPO_ID   = "east-right/bge-m3-cafe-finetuned"
DATASET_REPO_ID = "east-right/bge-m3-cafe-train-data"


def upload_model():
    print(f"📤 모델 업로드 중...")
    api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Update finetuned model",
    )
    print(f"✅ 모델 업로드 완료!")


def upload_dataset():
    print(f"\n📤 데이터셋 업로드 중...")
    api.create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", private=True, exist_ok=True)

    for file_path in [
        EMB_DATA_DIR / "finetune_train" / "train.jsonl",
        EMB_DATA_DIR / "test_query_pos.json",
    ]:
        if file_path.exists():
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                commit_message=f"Upload {file_path.name}",
            )
            print(f"  ✅ {file_path.name} 업로드 완료")

    print(f"✅ 데이터셋 업로드 완료!")


def link_dataset_to_model():
    print(f"\n🔗 모델 카드 업데이트 중...")
    model_card = f"""---
language:
- ko
license: apache-2.0
datasets:
- {DATASET_REPO_ID}
tags:
- sentence-transformers
- embeddings
- cafe-recommendation
- bge-m3
---

# BGE-M3 카페 추천 시스템 파인튜닝 모델

카페 추천 시스템의 Rule 검색 성능 향상을 위해 `BAAI/bge-m3`를 도메인 특화 파인튜닝한 모델입니다.

## 성능 (test set, 검색 대상 rule 180개)

| 지표 | BGE-M3-ko (기준) | 파인튜닝 후 | 향상 |
|------|:-:|:-:|:-:|
| Recall@1 | 0.281 | 0.545 | **+0.264 ↑** |
| Recall@5 | 0.519 | 0.822 | **+0.303 ↑** |
| Recall@10 | 0.619 | 0.888 | **+0.269 ↑** |
| Recall@20 | 0.679 | 0.962 | **+0.284 ↑** |

## 학습 데이터

- 데이터셋: [{DATASET_REPO_ID}](https://huggingface.co/datasets/{DATASET_REPO_ID}) (private)
- query-pos 2002쌍 → confidence stratify 80/20 split (seed=42)
- 메뉴-복합 증강(split 후 train/test 각각) + 하드 네거티브 7개/query

## 파인튜닝 설정

- 베이스 모델: `BAAI/bge-m3`
- Epoch: 5, Batch: 16, LR: 1e-5, Temperature: 0.02
- 하드 네거티브 7개/query (train_group_size 8) — 베이스 모델로 채굴
- GPU: NVIDIA A40, fp16
"""
    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Link dataset to model card",
    )
    print(f"✅ 모델 카드 업데이트 완료!")


def main():
    upload_model()
    upload_dataset()
    link_dataset_to_model()
    print(f"\n🎉 완료!")
    print(f"  모델:   https://huggingface.co/{MODEL_REPO_ID}")
    print(f"  데이터: https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()