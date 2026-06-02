# Data Preprocessing Branch — `data-preprocessing`

카페 추천 시스템(SOAR + XAI)을 위한 매장 item 생성 전처리 파이프라인입니다.  
리뷰 텍스트에서 NER 추출 → 키워드 정규화 → 매장별 대표 item 집계까지의 전 과정을 담고 있습니다.

---

## 전체 파이프라인 개요

```
리뷰 CSV
   ↓
[STEP 1] LLM 기반 NER 추출 (preprocesing_batch.py)
   ↓
[STEP 2] 키워드 정규화 (item_preprocessing.ipynb)
   ├── FACILITY / ATMOSPHERE / TARGET → 임베딩 클러스터링 + LLM 대표 키워드
   └── MENU → LLM 기반 메뉴명 정규화
   ↓
[STEP 3] 매장별 긍정 item 집계 (TF-IDF)
   ↓
[STEP 4] 매장별 부정 item 집계
   ↓
[STEP 5] 최종 매장 item 산출물 저장
```

---

## STEP 1. LLM 기반 NER 추출

**파일:** `preprocesing_batch.py`  
**산출물:** `data/ner_result.json`

OpenAI Batch API를 활용해 카페 리뷰에서 4가지 카테고리의 개체명을 추출합니다.

| 카테고리 | 정의 |
|---|---|
| MENU | 음식/음료 메뉴 |
| FACILITY | 매장의 물리적 시설 |
| ATMOSPHERE | 추상적 분위기/감성 |
| TARGET | 방문 목적/동반자/타겟 |

각 entity는 `entity`, `descriptor`, `sentiment` 3가지 필드로 구성됩니다.

**Batch API를 사용한 이유:**  
15,000개 리뷰를 실시간 API로 처리하면 약 24시간 이상 소요되고 비용도 높아집니다. OpenAI Batch API는 비동기로 처리되어 **비용 50% 절감** 및 rate limit 문제를 회피할 수 있습니다. 2,000건 단위 청크로 나눠 제출하고, 완료된 것은 재처리하지 않도록 멱등성을 보장했습니다.

---

## STEP 2. 키워드 정규화

**파일:** `item_preprocessing.ipynb`  
**산출물:** `data/keyword.csv`

### 2-1. FACILITY / ATMOSPHERE / TARGET 정규화

**문제:**  
NER 결과에는 "조용하다", "조용한", "조용함", "조용한 카페" 등 동일한 의미의 키워드가 다양한 표현으로 존재합니다.

**해결 방안:**  
임베딩 기반 클러스터링 + LLM 대표 키워드 추출의 2단계 접근법을 사용했습니다.

1. **BGE-M3-ko** 모델로 키워드를 임베딩합니다.  
   단순 단어 임베딩보다 `"이 카페의 핵심 특징은 {kw}입니다."` 형태로 도메인 문장을 확장해 임베딩하면 카페 문맥이 반영되어 클러스터링 품질이 향상됩니다.
2. **UMAP + HDBSCAN**으로 유사 키워드를 군집화합니다.
3. 각 군집을 **GPT-4o-mini**에 전달해 대표 키워드를 선정합니다.
4. 1차 클러스터링에서 노이즈(-1)로 분류된 키워드는 **재군집 후 2차 LLM 처리**합니다.
5. 최종적으로 `원본_키워드 → 퍼지매칭_키워드` 매핑 딕셔너리(`df_fuzzy`)를 생성합니다.

### 2-2. MENU 키워드 정규화

**문제:**  
메뉴는 카페마다 고유한 이름이 많아 임베딩 클러스터링으로는 정규화가 어렵습니다.  
예: "수박파이"와 "수박주스"가 벡터 공간에서 가깝게 위치해 오분류될 수 있습니다.  
또한 "아아", "얼죽아" 같은 한국 카페 문화 특유의 줄임말은 현실 세계 지식이 필요합니다.

**해결 방안:**  
**GPT-5.1**을 활용해 현실 세계 지식 기반으로 메뉴명을 정규화했습니다.

```
아아            → 아이스아메리카노  (줄임말 확장)
아이스라떼       → 라떼             (온도 수식어 제거)
인생베이글       → 베이글           (매장 고유 브랜딩 제거)
밤티라미수       → 티라미수         (재료 수식어 제거)
초코스콘        → 초코스콘          (재료가 메뉴 구분 → 유지)
커피, 음료 등   → 원문 유지         (카테고리 포괄어)
맛, 원두 등     → null             (비메뉴 키워드 필터링)
```

수식어 제거 기준은 **수식어 종류**로 판단합니다. 온도/강도/스타일/시즌 수식어는 제거하고, 재료 수식어는 메뉴를 구분짓는 경우에만 유지합니다.

---

## STEP 3. 매장별 긍정 item 집계 (TF-IDF)

**파일:** `item_preprocessing.ipynb`  
**산출물:** `df_tfidf_all` → `data/market_item.csv` (긍정 파트)

### 집계 방식

**문제:**  
단순 빈도(count)로 집계하면 리뷰 수가 많은 매장이 유리하고, 모든 카페에 공통적으로 나오는 "카페 분위기", "커피_맛있다" 같은 키워드가 상위를 독점합니다.

**해결 방안:**  
2가지 문제를 각각의 방법으로 해결했습니다.

**리뷰 수 불균형 문제:**  
매장별 긍정 키워드 수를 기준으로 threshold(10개)를 설정합니다.
- 키워드 수 ≥ 10 → **비율(%)** 기준 Top 10
- 키워드 수 < 10 → **count** 기준 Top 10

**공통 키워드 독점 문제:**  
Top 10 키워드를 대상으로 **TF-IDF**를 적용해 타 매장 대비 차별적인 키워드를 선별합니다.
- TF: 해당 매장 내 키워드 등장 비율
- IDF: 전체 매장 중 해당 키워드가 등장한 매장 수의 역수
- 결과: 키워드타입별 TF-IDF 상위 **5개** 선정

### item 생성 방식

키워드타입에 따라 item 생성 방식을 다르게 적용합니다.

| 키워드타입 | item 생성 방식 | 이유 |
|---|---|---|
| MENU | `키워드_descriptor` 결합 | descriptor(맛 표현)가 구체적인 정보를 담고 있음 |
| FACILITY | `키워드_descriptor` 결합 | descriptor(상태 표현)가 시설의 특성을 설명함 |
| ATMOSPHERE | 키워드 그대로 | 키워드 자체가 이미 형용사형으로 의미를 내포 |
| TARGET | 키워드 그대로 | 방문 목적 자체가 명확한 의미 단위 |

같은 키워드에 descriptor가 여러 개인 경우, 가장 많이 등장한 descriptor 하나만 선택합니다.  
MENU 포괄어(커피, 음료, 디저트, 빵, 푸드, 주류)는 집계에서 제외합니다.

---

## STEP 4. 매장별 부정 item 집계

**파일:** `item_preprocessing.ipynb`  
**산출물:** `df_neg_final` → `data/market_item.csv` (부정 파트)

**부정 키워드에 TF-IDF를 사용하지 않은 이유:**  
TF-IDF는 "희귀할수록 중요하다"는 가정에 기반합니다. 그러나 부정 정보는 차별점이 아니라 **있냐 없냐**가 중요합니다. 화장실 불결함은 1개 매장에만 있어도 중요한 경고이며, 희귀하다고 더 중요한 것이 아닙니다.

**동점 처리 방식:**  
부정 키워드는 수가 적어 count 동점이 많이 발생합니다. 동점 시 **전체 매장 기준 global_count**가 높은 키워드를 우선합니다. 더 많은 매장에서 공통적으로 지적된 단점이 더 공공연한 문제이기 때문입니다.

**필터링 기준:**
- 매장 내 count = 1 → 제외 (단발성 불만)
- 전체 global_count = 1 → 제외 (매우 희귀한 불만)

**집계 대상:** MENU, FACILITY, ATMOSPHERE (TARGET은 데이터 수가 너무 적어 제외)

---

## STEP 5. 최종 매장 item 저장

**파일:** `item_preprocessing.ipynb`  
**산출물:** `data/market_item.csv`

긍정 item(df_tfidf_all)과 부정 item(df_neg_final)을 합쳐 최종 매장 item 데이터를 생성합니다.

**최종 컬럼 구성:**

| 컬럼 | 설명 |
|---|---|
| 사업장명 | 매장명 |
| 키워드타입 | MENU / FACILITY / ATMOSPHERE / TARGET |
| sentiment | 긍정 / 부정 |
| 최종_키워드 | 정규화된 대표 키워드 |
| tfidf_score | TF-IDF 점수 (긍정만) |
| count | 매장 내 등장 횟수 (부정만) |
| global_count | 전체 매장 기준 등장 횟수 (부정만) |

**매장당 item 구성:**
- 긍정: 키워드타입별 최대 5개 × 4타입 = 최대 20개
- 부정: 키워드타입별 최대 3개 × 3타입 = 최대 9개

---
## 번외. gliner을 활용한 ner tagging
- 한국어로 파인튜닝된 모델을 사용하였지만 결과가 매우 미흡
- 기본적으로 파인튜닝에 사용된 데이터가 문어체 데이터에 한정되어 있어 결과가 아쉬움
- 만약 실시간 성을 높혀 사용하고 싶으면 높은 수준의 파인튜닝이 함께 시행되어야 함
---

## 파일 구조

```
data-preprocessing/
├── __pycache__/
├── gliner_test.ipynb           # GliNER 모델 테스트 (초기 실험)
├── item_preprocessing.ipynb    # 키워드 정규화 + item 집계 (메인)
├── preprocesing_batch_while.py # 배치 자동 반복 처리
├── preprocesing_batch.py       # NER 배치 처리 (메인)
├── preprocesing.py             # 단건 처리 버전
├── prompt.py                   # system_prompt, cluter_system_prompt,
│                               # MENU_NORMALIZE_SYSTEM_PROMPT
└── README.md

data/
├── batch_id.txt                     # 배치 제출 ID 저장
├── cafe_questions.json              # 카페 질문 데이터
├── keyword.csv                      # 정규화된 키워드 (df_final)
├── market_item.csv                  # 최종 매장 item (산출물) ★
├── ner_keyword_sent_clt.csv         # 1차 클러스터링 결과
├── ner_keyword_sent_re_clt.csv      # 재군집 결과
├── ner_keyword_word_clt.csv         # 단어 기반 클러스터링 결과
├── ner_result.json                  # NER 추출 결과
├── seoul_restaurants_shinline.csv   # 원본 매장 데이터
├── shinline_cafe_ids_v2.csv         # 카페 ID 목록
└── shinline_cafe_reviews_test.csv   # 원본 리뷰 데이터 (입력)
```

---

## 주요 기술 스택

| 항목 | 내용 |
|---|---|
| 임베딩 모델 | `dragonkue/BGE-m3-ko` (한국어 특화) |
| 차원 축소 | UMAP |
| 클러스터링 | HDBSCAN |
| LLM (클러스터 대표) | GPT-4o-mini |
| LLM (메뉴 정규화) | GPT-5.1 |
| LLM (NER) | GPT-4o-mini (Batch API) |
| 키워드 가중치 | 수동 구현 TF-IDF |