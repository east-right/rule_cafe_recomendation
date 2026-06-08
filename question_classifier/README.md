# Question Classifier

카페 추천 시스템의 질문 유효성 검증 모델입니다.  
사용자 질문을 4가지 클래스로 분류하여 적절한 처리 경로로 라우팅합니다.

카페 추천 시스템 아키텍처에서 추천의 방식은 촐 3가지가 존재합니다.
1. 비메뉴 관련 추천
    - 해당 부분은 리뷰 데이터의 키워드를 활용하여 itent에 맞는 Rule에 기반해 추천합니다.
2. 메뉴 관련 추천
    - 해당 부분은 리뷰 데이터의 키워드를 활용하여 매장을 필터링 후 메뉴 전용 Rule에 기반해 추천합니다.
3. 메뉴+비메뉴 관련 추천
    - Rule은 비메뉴에 기반하여 추천하고 메뉴에 기반한 초기 매장 필터링을 거칩니다.
4. Fallback
    - 해당 질문이 들어오면 Fallback 시킵니다. 유효하지 않은 질문이라고 답변하도록 최종 모델에게 유도하는 prompt를 제공합니다.

---

## 분류 클래스

| 클래스 | 설명 | 예시 |
|---|---|---|
| `non-menu` | 메뉴 외 속성 질문 | "조용하고 콘센트 있는 카페 추천해줘" |
| `menu-only` | 메뉴만 묻는 질문 | "아메리카노 맛있는 카페 어디야?" |
| `menu-complex` | 메뉴 + 비메뉴 복합 질문 | "아메리카노 맛있으면서 조용한 카페 추천해줘" |
| `invalid` | 카페와 무관한 질문 | "북태평양 기단이 머무르는 기간은?" |

---

## 파일 구조

```
question_classifier/
├── config/
│   ├── roberta_base.yaml       # klue/roberta-base 학습 설정
│   └── roberta_large.yaml      # klue/roberta-large 학습 설정 (미사용)
├── data/
│   ├── menu_complex.jsonl      # GPT-4.1-mini 생성 데이터 (600개)
│   ├── invalid.jsonl           # KLUE-MRC 샘플링 데이터 (600개)
│   ├── train.jsonl             # 학습 데이터 (2123개)
│   └── test.jsonl              # 평가 데이터 (236개)
├── build_dataset.py            # 전체 데이터 합치고 train/test 분할
├── evaluate.py                 # 모델 평가
├── generate_invalid.py         # KLUE-MRC에서 invalid 추출
├── generate_menu_complex.py    # GPT로 menu-complex 생성
├── prompt.py                   # GPT 생성용 / 파인튜닝용 프롬프트 관리
├── train.py                    # RoBERTa 파인튜닝
├── upload.py                   # HuggingFace 업로드
└── pyproject.toml              # 파인튜닝 전용 의존성
```

---

## 데이터 구축

### 클래스별 데이터 소스

| 클래스 | 소스 | 개수 |
|---|---|---|
| `non-menu` | `valid_questions.json` (single+multi) 샘플링 | 600개 |
| `menu-only` | `valid_questions.json` (menu) 전체 | 559개 |
| `menu-complex` | GPT-4.1-mini 배치 생성 | 600개 |
| `invalid` | KLUE-MRC 데이터셋 샘플링 | 600개 |

### menu-complex 생성 방식
`market_item.csv`의 메뉴 키워드(910개) + `valid_questions.json`의 비메뉴 키워드(462개)를 랜덤 조합하여 GPT-4.1-mini에게 자연스러운 질문 생성 요청. 배치당 10개씩 처리.

```
메뉴 키워드: 아메리카노
비메뉴 키워드: 조용한 카페, 콘센트_있다
→ "아메리카노 맛있으면서 조용하고 콘센트도 있는 카페 추천해줘"
```

### invalid 생성 방식
KLUE-MRC(뉴스 기사 기반 한국어 QA) 데이터셋에서 카페와 무관한 질문 600개 랜덤 샘플링.

### train/test 분할
클래스별 stratified 9:1 분할 (train: 2123개 / test: 236개)

---

## 모델 선정

### 베이스 모델: klue/roberta-base
- 한국어 특화 RoBERTa 모델
- 텍스트 분류 태스크에 최적화된 구조 ([CLS] 토큰 → 분류 헤드)
- `klue/roberta-large`도 검토했으나 디스크 용량 문제로 base로 확정
- base 모델만으로도 충분한 성능 달성

### 파인튜닝 방식
- **풀 파인튜닝** (LoRA/QLoRA 불필요 - 분류 헤드만 추가하는 경량 구조)
- GPU: NVIDIA A40 (RunPod)
- fp16 혼합 정밀도 학습

---

## 학습 설정

```yaml
model_name: "klue/roberta-base"
epochs: 5
batch_size: 32
learning_rate: 2e-5
max_seq_length: 128
metric_for_best_model: f1_macro
```

---

## 평가 결과

```bash
python question_classifier/evaluate.py --config question_classifier/config/roberta_base.yaml
```

| 지표 | 값 |
|---|---|
| 전체 Accuracy | **0.9788** |
| F1 (macro) | **0.9790** |
| 평균 추론 시간 | **0.0076s/건** |

### 클래스별 성능

| 클래스 | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| non-menu | 0.97 | 0.95 | 0.96 | 60 |
| menu-only | 0.98 | 1.00 | 0.99 | 56 |
| menu-complex | 0.97 | 0.97 | 0.97 | 60 |
| invalid | 1.00 | 1.00 | 1.00 | 60 |

→ invalid 완벽 분류. non-menu/menu-complex 경계 케이스에서 소수 오분류 발생 (데이터 자체의 모호성).

---

## HuggingFace

| 항목 | 링크 |
|---|---|
| 질문 분류 모델 | east-right/cafe-question-classifier-roberta-base (private) |

---

## RunPod 실행

```bash
git clone -b feat/question-classifier https://github.com/east-right/cafe_recomendation.git /workspace/cafe_recomendation
cd /workspace/cafe_recomendation
echo "HUGGINGFACE_TOKEN_WRITE=your_token" > .env
cd question_classifier
uv sync
source .venv/bin/activate
python train.py --config config/roberta_base.yaml
python evaluate.py --config config/roberta_base.yaml
python upload.py --config config/roberta_base.yaml
```

---

## .env 설정

```dotenv
OPENAI_API_KEY=sk-xxx
HUGGINGFACE_TOKEN_READ=hf_xxx
HUGGINGFACE_TOKEN_WRITE=hf_xxx
```