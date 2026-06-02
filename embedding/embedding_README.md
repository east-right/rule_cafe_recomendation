# Embedding Branch — `feat/keyword-select`

카페 추천 시스템의 Rule 검색 성능 향상을 위한 BGE-M3 임베딩 모델 파인튜닝 파이프라인입니다.

---

## 개요

사용자 질문이 들어왔을 때 OpenSearch RAG를 통해 적절한 질문 키워드(SOAR Rule)를 선택하는 구조에서, 임베딩 모델의 검색 성능을 향상시키기 위해 도메인 특화 파인튜닝을 진행했습니다.

```
질문 입력
    ↓
임베딩 모델 (BGE-M3 파인튜닝)
    ↓
OpenSearch RAG → Rule 후보 10개
    ↓
sLLM → 최적 Rule 선택
    ↓
SOAR 실행 → 매장 추천
```

---

## 파인튜닝 결과

| 지표 | BGE-M3-ko (기준) | 파인튜닝 후 | 향상 |
|------|:-:|:-:|:-:|
| Recall@1 | 0.3509 | 0.5307 | **+0.1798 ↑** |
| Recall@5 | 0.5395 | 0.8114 | **+0.2719 ↑** |
| Recall@10 | 0.6184 | 0.8728 | **+0.2544 ↑** |

- 평가 데이터: test set 228개 (전체 1136개의 20%)
- 비교 모델: `dragonkue/BGE-m3-ko`
- 파인튜닝 모델: HuggingFace에 업로드

---

## 파일 구조

```
embedding/
├── data/
│   ├── finetune_train/
│   │   └── train.jsonl        # 학습 데이터 (908개)
│   └── test_query_pos.json    # 평가 데이터 (228개)
├── models/
│   └── bge-m3-cafe/           # 파인튜닝된 모델(비공개 처리)
├── finetune_bge.py            # 파인튜닝 스크립트
├── evaluate.py                # 성능 평가 스크립트
├── requirements.txt           # 의존 패키지
└── README.md
```

---

## 학습 데이터 구성

```
data/rule_metadata_merged.json (313개 rule)
    ↓
각 rule의 question + merged_questions → query
각 rule의 description → positive
랜덤 다른 rule description → negative (easy negative)
    ↓
전체 1136개 query-pos-neg 쌍
train: 908개 (80%) / test: 228개 (20%), seed=42
```

---

## 파인튜닝 설정

| 항목 | 값 |
|------|-----|
| 베이스 모델 | `BAAI/bge-m3` |
| Epoch | 3 |
| Batch Size | 16 |
| Learning Rate | 1e-5 |
| Temperature | 0.02 |
| Max Length | 512 |
| Loss | InfoNCE |
| GPU | NVIDIA A40 |

---

## 실행 방법

### 환경 설정

```bash
pip install -r embedding/requirements.txt
```

### 파인튜닝

```bash
torchrun --nproc_per_node=1 embedding/finetune_bge.py
```

### 평가

```bash
python embedding/evaluate.py
```

---

## 의존 패키지

```
FlagEmbedding==1.4.0 
transformers==4.47.0
peft==0.14.0
accelerate
huggingface_hub
sentencepiece
numpy
```

## 피드백
- 데이터 전처리 퀄리티가 떨어져 하드 네거티브 데이터셋 구축 사용 부적합
    - 오히려 결과를 더 나쁘게 만들 위험 야기
- 좋은 데이터셋으로 파인튜닝 시 dense 버전 보단 sparse 사용 예정