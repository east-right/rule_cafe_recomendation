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
| Recall@1 | 0.3017 | 0.5237 | **+0.2220 ↑** |
| Recall@5 | 0.5387 | 0.8279 | **+0.2892 ↑** |
| Recall@10 | 0.6284 | **0.9077** | **+0.2793 ↑** |

- 검색 대상: rule 180개 (single 97 + multi 83)
- 평가 데이터: test set 401개 (전체 2002개의 20%)
- 비교 모델: `dragonkue/BGE-m3-ko`
- 파인튜닝 모델: HuggingFace에 업로드
- 하드 네거티브 7개/query 채굴 적용 (v1 대비 Recall 향상)

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
data/rule_metadata_merged.json (180개 rule)
    ↓
각 rule의 question + merged_questions → query
각 rule의 description → positive
베이스 모델로 채굴한 헷갈리는 rule description → hard negative (7개/query)
    ↓
전체 2002개 query-pos 쌍
train: 1601개 (80%) / test: 401개 (20%), seed=42
```

---

## 파인튜닝 설정

| 항목 | 값 |
|------|-----|
| 베이스 모델 | `BAAI/bge-m3` |
| Epoch | 5 |
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

## v2 변경점
- 데이터 정규화(v2 market_item) 후 rule 셋 재생성 → 검색 대상 180개 (single 97 + multi 83)
- multi rule 순열 중복 통합 + 약한 조합 제거로 검색 공간 sparse화 → Recall 상승
- **하드 네거티브 채굴 적용** (v1에선 데이터 품질 문제로 보류했던 것):
  베이스 모델로 query별 가장 헷갈리는 rule을 neg로 채굴(7개) → 미세 분리 학습
  → Recall@10 0.87 → **0.91** 돌파

## 남은 과제
- rule description이 다소 generic("~분들을 위한 추천")해서 유의어 rule 간 1위 경쟁 발생
  (R@1 0.52) → 최종 선택은 후속 sLLM 단계가 담당
- 검색 검사: `inspect_retrieval.py`로 오답 케이스 확인 가능
