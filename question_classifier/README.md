# Question Classifier

카페 추천 시스템의 질문 유효성 검증 모델입니다.  
사용자 질문을 4가지 클래스로 분류하여 적절한 처리 경로로 라우팅합니다.

---

## 분류 클래스

| 클래스 | 설명 | 처리 방향 |
|---|---|---|
| `non-menu` | 메뉴 외 속성 질문 | RAG → sLLM rule 선택 |
| `menu-only` | 메뉴만 묻는 질문 | 메뉴 키워드 추출 → 매장 필터링 (미구현) |
| `menu-complex` | 메뉴 + 비메뉴 복합 질문 | RAG → sLLM rule 선택 + 메뉴 필터링 (미구현) |
| `invalid` | 카페와 무관한 질문 | fallback 처리 |

---

## 파일 구조

```
question_classifier/
├── config/
│   ├── roberta_base.yaml       # klue/roberta-base 설정
│   ├── roberta_large.yaml      # klue/roberta-large 설정
│   └── koelectra_base.yaml     # koelectra-base-v3 설정 (최종 선택)
├── data_augment/               # 데이터 생성/증강
│   ├── data/
│   │   ├── menu_complex.jsonl  # GPT 생성 (600개)
│   │   ├── invalid.jsonl       # GPT 생성 (600개)
│   │   ├── augmented.jsonl     # 스타일 augmentation (1025개)
│   │   ├── hard_test.jsonl     # Hard test 데이터 (300개)
│   │   ├── trainval.jsonl      # 학습/검증 데이터 (3149개)
│   │   └── test.jsonl          # 최종 평가 데이터 (235개)
│   ├── build_dataset.py        # 전체 데이터 합치고 분할
│   ├── generate_menu_complex.py
│   ├── generate_invalid.py
│   ├── generate_augmented.py
│   ├── generate_hard_test.py
│   └── prompt.py               # GPT 생성용 프롬프트 관리
└── finetune/                   # 파인튜닝
    ├── config/ → ../config/
    ├── train.py
    ├── evaluate.py
    ├── evaluate_prefix.py
    ├── upload.py
    ├── predict.py
    └── pyproject.toml
```

---

## 데이터 구축

### 클래스별 소스

| 클래스 | 소스 | 개수 |
|---|---|---|
| `non-menu` | `valid_questions.json` (single+multi) 샘플링 | 600개 |
| `menu-only` | `valid_questions.json` (menu) 전체 | 559개 |
| `menu-complex` | GPT-4.1-mini 배치 생성 | 600개 |
| `invalid` | GPT-4.1-mini 배치 생성 | 600개 |

### menu-complex 생성
`market_item.csv`의 **MENU 타입** 키워드(438개) + `valid_questions.json`의 비메뉴 키워드(462개) 조합 → GPT 배치 생성 (10개/배치)

> 주의: 초기 버전에서 `market_item.csv` 전체(MENU+ATMOSPHERE+TARGET+FACILITY)를 사용해 약 70%가 비메뉴 키워드로 생성되는 버그 발생 → `키워드타입 == 'MENU'` 필터링으로 수정

### invalid 생성
유형별 GPT 생성:
- **유형1 (40%)**: 카페 외 업종 추천 ("식당 추천해줘", "술집 어디야")
- **유형2 (35%)**: 카페 관련이지만 추천 불가 ("카페 몇시에 열어?", "와이파이 비번")
- **유형3 (25%)**: 완전 무관 ("오늘 날씨 어때?", "영화 추천해줘")

> 초기 버전에서 KLUE-MRC(뉴스 기사 기반) 사용 시 일상적 invalid 케이스 처리 불가 확인 → GPT 생성으로 전환

### 스타일 Augmentation
기존 trainval 데이터를 다양한 스타일로 변환 (클래스당 80개 샘플 x 3개 변환 = 1025개 추가):
- 구어체/반말: "어디야", "없나", "알려줘ㅋㅋ"
- 줄임말: "아아", "뜨아", "카공"
- 불완전한 짧은 문장: "조용한데", "콘센트 있는 카페"
- 감탄사/이모티콘: "ㅠㅠ", "~"
- 영어 혼용: "wifi 빵빵한 카페"

### Hard Test 데이터 (별도)
분류하기 어려운 엣지 케이스 300개 (클래스당 75개):
- 간접적 표현: "오래 앉아도 눈치 안 주는 곳"
- 줄임말 메뉴: "아아 맛있는 카페"
- 경계 케이스: "카페인데 공부하기 좋은 분위기인가?"

### 데이터 분할
클래스별 stratified 9:1 (trainval/test), trainval은 5-fold 교차검증

| 구분 | 개수 |
|---|---|
| trainval | 3149개 (augmentation 포함) |
| test | 235개 |
| hard_test | 300개 (별도) |

---

## 모델 선정

### 비교 실험

| 모델 | 일반 Test Acc | Hard Test Acc | Hard Test F1 | 추론 시간 |
|---|---|---|---|---|
| klue/roberta-base (augmentation 전) | 0.9957 | 0.7233 | 0.7370 | 0.0076s |
| klue/roberta-base (augmentation 후) | 0.9872 | 0.8000 | 0.8064 | 0.0134s |
| **koelectra-base-v3 (최종)** | **0.9957** | **0.8000** | **0.8084** | **0.0066s** |

→ koelectra가 일반 Test 성능, Hard Test F1, 추론 속도 모두 우수하여 최종 선택

### 파인튜닝 방식
- **풀 파인튜닝** (분류 헤드 추가, 경량 구조라 LoRA 불필요)
- **5-Fold 교차검증**으로 데이터 robustness 검증
- GPU: NVIDIA A40 (RunPod)

---

## 5-Fold 교차검증 결과 (koelectra-base-v3)

| Fold | train_loss | val_loss | Accuracy | F1 |
|---|---|---|---|---|
| 1 | 0.2713 | 0.0284 | 0.9937 | 0.9937 |
| 2 | 0.2991 | 0.0601 | 0.9810 | 0.9809 |
| 3 | 0.3069 | 0.0811 | 0.9810 | 0.9810 |
| 4 | 0.2905 | 0.0456 | 0.9905 | 0.9904 |
| 5 | 0.2959 | 0.0515 | 0.9905 | 0.9905 |
| **평균** | **0.2927** | **0.0533** | **0.9873 ± 0.0053** | **0.9873 ± 0.0053** |

train_loss > val_loss → 오버피팅 없음 ✅

---

## 최종 평가 결과

```bash
python evaluate.py --config ../config/koelectra_base.yaml
```

| 구분 | Accuracy | F1 (macro) | 평균 추론 시간 |
|---|---|---|---|
| 일반 Test (235개) | 0.9957 | 0.9956 | 0.0066s/건 |
| Hard Test (300개) | 0.8000 | 0.8084 | 0.0066s/건 |

### 클래스별 성능 (Hard Test)

| 클래스 | Precision | Recall | F1 |
|---|---|---|---|
| non-menu | 0.58 | 0.87 | 0.70 |
| menu-only | 0.98 | 0.80 | 0.88 |
| menu-complex | 0.97 | 0.87 | 0.92 |
| invalid | 0.83 | 0.67 | 0.74 |

### 한계
- 간접적 표현 ("오래 앉아도 눈치 안 주는 곳") → invalid 오분류
- 줄임말 메뉴 ("아이스아메리카노 맛집") → 학습 데이터 미포함 시 오분류
- 데이터 추가 보강 예정

---

## OpenAI 교체 (gpt-4.1-mini)

서빙 단계에서 koelectra 파인튜닝 모델을 **OpenAI `gpt-4.1-mini`로 교체**했다.
GPU 호스팅·재학습 부담 없이 더 높은 일반화 성능을 얻기 위함이며, 아래 **동일 hard_test(300개)·동일 라벨 기준** 비교로 교체 근거를 확인했다.

### Hard Test 비교 (동일 300개, 동일 라벨)

| 모델 | Accuracy | F1 (macro) | 추론 시간/건 |
|---|---|---|---|
| koelectra-base-v3 | 0.8000 | 0.8084 | 0.0066s |
| **gpt-4.1-mini** | **0.9400** | **0.9402** | 0.118s |

→ **Acc +14%p, F1 +13%p.** koelectra의 약점 클래스가 크게 개선됨:

| 클래스 (F1) | koelectra | gpt-4.1-mini |
|---|---|---|
| non-menu | 0.70 | **0.91** |
| menu-only | 0.88 | 0.96 |
| menu-complex | 0.92 | 0.97 |
| invalid | 0.74 | **0.92** |

- **트레이드오프**: 추론 시간 0.0066s → 0.118s (약 18배). 챗봇 단건 응답엔 무방.
- **비용**: 호출당 OpenAI API (gpt-4.1-mini, 프롬프트 짧아 건당 ≈ $0.0001).
- **재현**: `uv run python question_classifier/evaluate_openai.py` (오답 18개 → `hard_test_openai_wrong.jsonl`).

> 구현 `classify.py` · 평가 `evaluate_openai.py` — koelectra `finetune/evaluate.py`와 동일 데이터·라벨 기준.

---

## HuggingFace

| 항목 | 링크 |
|---|---|
| 질문 분류 모델 | east-right/cafe-question-classifier-koelectra-base (private) |

---

## RunPod 실행

```bash
git clone -b feat/question-classifier https://github.com/east-right/cafe_recomendation.git /workspace/cafe_recomendation
cd /workspace/cafe_recomendation
echo "HUGGINGFACE_TOKEN_WRITE=your_token" > .env
cd question_classifier/finetune
uv sync
source .venv/bin/activate
python train.py --config ../config/koelectra_base.yaml
python evaluate.py --config ../config/koelectra_base.yaml
python upload.py --config ../config/koelectra_base.yaml
```

---

## .env 설정

```dotenv
OPENAI_API_KEY=sk-xxx
HUGGINGFACE_TOKEN_READ=hf_xxx
HUGGINGFACE_TOKEN_WRITE=hf_xxx
```