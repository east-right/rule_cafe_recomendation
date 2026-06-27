# Data Preprocessing — `data-normalization-v2`

카페 리뷰에서 **매장 item**(특징·메뉴 평가)을 생성하는 전처리 파이프라인입니다.
리뷰 텍스트 → NER → 키워드 정규화/클러스터링 → 매장별 item 집계까지의 전 과정을 담습니다.

> v1 산출물·스크립트는 `archive/`, `../data/archive/`에 보관되어 있습니다.

---

## 전체 파이프라인

```
리뷰 CSV (../data/shinline_cafe_reviews_test.csv)
   │
[1] 2-pass NER 추출            preprocesing_batch_while.py  → ner_result_v2.json
   │
[2] 정규화 + 개념 클러스터링    normalize.py                 → ner_result_v3.json
   │
   ├─[3] 비메뉴 매장 item 집계   market_aggregate.py          → market_item.csv
   └─[4] MENU 3계층 평가 테이블  menu_aggregate.py            → menu_item.csv
```

실행:
```bash
uv run python preprocesing_batch_while.py   # [1] OpenAI Batch API (장시간)
uv run python normalize.py                  # [2] LLM 정규화/클러스터링
uv run python market_aggregate.py           # [3] 계산만 (LLM 없음)
uv run python menu_aggregate.py             # [4] LLM 분류 + 집계
```

---

## STEP 1. 2-pass NER 추출

**파일:** `preprocesing_batch_while.py` · `prompt.py`
**산출물:** `../data/ner_result_v2.json`

리뷰를 2단계로 분석해 개체(entity)와 평가(descriptor)를 분리 추출합니다.

- **Pass 1** — `entity` / `category` / `type` / `sentiment`
- **Pass 2** — EVALUATIVE entity에만 `descriptor` 부여 (+ confidence)

| 필드 | 설명 |
|---|---|
| category | MENU / FACILITY / ATMOSPHERE / TARGET |
| type | EVALUATIVE(평가 대상) / FEATURE(존재 사실) |
| sentiment | 긍정 / 부정 / 중립 |
| descriptor | EVALUATIVE에만. 평가 표현(맛있다, 넓다 등) |
| review_idx, source_review | 원본 리뷰 매핑 |

**왜 2-pass인가:** v1은 `entity`에 형용사가 섞여("조용한 카페") 후속 정규화가 어려웠습니다. entity와 descriptor를 분리하면 정규화 단위가 명확해지고, EVALUATIVE에만 descriptor를 요청해 노이즈를 줄입니다. confidence="애매"는 후처리에서 None 처리.

**Batch API:** gpt-4.1-mini Batch API로 비용 50%↓·rate limit 회피. 카페 단위 청킹(5리뷰/청크, 최대 150리뷰/카페)으로 매장 누락 방지. 할당량 초과 시 지수 백오프 재시도, 완료 청크는 재처리 안 함(멱등).

산출 규모: 201개 카페 / entity 23,825개 (descriptor 커버리지 94.4%).

---

## STEP 2. 정규화 + 개념 클러스터링

**파일:** `normalize.py`
**산출물:** `../data/ner_result_v3.json`, `../data/norm_mapping/*.json`

표면형이 제각각인 키워드를 일관된 대표어로 통일합니다. 계층적으로 적용:

```
[1] 표기 정규화 (규칙)        공백/대소문자 — 비용 0
[3] LLM 규칙 정규화           카테고리별 정책 (아이스아메리카노→아메리카노 등)
[4] descriptor 정규화         동의어 병합 + 극성 보존(맛있다≠맛없다) + 복합→주형용사
[4.5] 공백-무시 병합          수박주스/수박 주스 → 빈도 최다형으로 통일
[4.7] 개념 클러스터링         비-MENU 한정. 잔챙이를 개념으로 (온라인 클러스터링)
[5] 매핑 적용
```

### 개념 클러스터링 (비-MENU)

FACILITY/ATMOSPHERE/TARGET의 롱테일(빈도 1~2 잔챙이)을 **일관된 개념**으로 묶습니다.
빈도 높은 순으로 청크 처리하며, 빈출 특징이 먼저 anchor가 되고 잔챙이가 흡수됩니다.

```
아침   ← 모닝커피, 아점, 아침에먹기      운동·산책 ← 산책, 등산, 따릉이, 다이어트
서울대 ← 서울대내, 서울대안, 캠퍼스내    콘센트   ← 충전기, 플러그, 멀티탭
```

- **변별 특징은 분리 유지**: 콘센트≠와이파이, 1인좌석≠단체석, 통창≠창문
- **추상 상위개념 금지**: '작업편의' 같은 뭉뚱그리기 X (구체적 개념만)

unique 감소: TARGET 415→66, ATMOSPHERE 167→79, FACILITY 391→140.

> **MENU는 클러스터링하지 않음.** 롱테일이 변형이 아니라 *고유 시그니처 메뉴*(체리블라썸, 말렌카 등)라, 유사도/개념 병합 시 시그니처가 파괴됨(수박파이/수박주스 오병합 위험). MENU는 STEP 1의 rule 정규화(온도 수식어 제거 등)까지만 적용.

### 매핑 파일 (`../data/norm_mapping/`)

| 파일 | 내용 |
|---|---|
| `entity_*.json` / `desc_*.json` | LLM 규칙 정규화 매핑 |
| `cluster_*.json` / `vocab_*.json` | 개념 클러스터 매핑 / 대표어휘 |
| `menu_class.json` | MENU specific → category/type 분류 |

---

## STEP 3. 비메뉴 매장 item 집계

**파일:** `market_aggregate.py`
**산출물:** `../data/market_item.csv`

FACILITY/ATMOSPHERE/TARGET을 매장 추천 근거(character item)로 집계합니다.

- **entity 단위 집계** + 대표 descriptor를 **metadata 컬럼**으로
  - `분위기_차분하다`처럼 키에 붙이지 않음 → 같은 `분위기`가 descriptor별로 쪼개지는 것 방지
- **긍정**: TF-IDF 상위 5개/카테고리 (흔한 `분위기`는 IDF로 자동 강등, count≥2 우선)
- **부정**: count 상위 3개/카테고리 (TARGET 제외, 동점 시 global_count)

TF-IDF를 쓰는 이유: 모든 카페에 흔한 키워드(분위기 좋다)를 누르고 **변별력 있는 특징**을 띄우기 위함.

**컬럼:** `place_id, 사업장명, category, 키워드, sentiment, 대표descriptor, count, tfidf_score, global_count`

---

## STEP 4. MENU 3계층 평가 테이블

**파일:** `menu_aggregate.py`
**산출물:** `../data/menu_item.csv`

MENU는 추천 파이프라인에서 **독립적으로 조회**되므로, 집계로 누르지 않고 **계층 테이블**로 보존합니다.
(`ner_result_v2.json` 원본에서 specific을 살려 사용 — v3는 type 레벨로 합쳐져 있어 사용 안 함)

| 레벨 | 예시 | 용도 |
|---|---|---|
| category | 커피/음료/디저트/베이커리/술/푸드 | "디저트 맛집?" |
| type | 아메리카노 / 케이크 / 빵 | "아메리카노 맛집?" |
| specific | 아이스아메리카노 / 쑥치즈케이크 | "이 집 시그니처?" |

### 맛집 점수

"판매중"(행 존재)과 "맛집"(잘함)을 구분하기 위한 지표:

```
맛집_score = (긍정 - 부정) × IDF
조건: 긍정비율 ≥ 0.6,  긍정 ≥ 2
IDF  = log(전체매장수 / 그 메뉴를 파는 매장수)
```

- **흔한 메뉴(아메리카노)**: IDF 낮음 → 압도적 긍정량이어야 맛집
- **희귀 시그니처(빨미까레)**: IDF 높음 → 긍정 몇 개로도 시그니처
- **부정 반영**: `(긍정-부정)` + 긍정비율 게이트로 호불호 메뉴 컷
- **약점 메뉴**: 부정 우세(`약점여부=Y`)로 별도 표시

**컬럼:** `place_id, 사업장명, category, type, specific, 총언급, 긍정, 부정, 중립, 대표descriptor, 맛집score, 약점여부`

---

## 검증 — 골든셋

**파일:** `golden_set_gen.py` → `../data/golden_set.csv`

정규화 품질을 측정하기 위한 손라벨 샘플(빈도 계층 추출). `canonical` 컬럼을 채워 자동 매핑과 비교.

---

## 파일 구조

```
data_preprocessing/
├── preprocesing_batch_while.py   # [1] 2-pass NER (Batch API)
├── prompt.py                     #     Pass1/Pass2 프롬프트
├── normalize.py                  # [2] 정규화 + 개념 클러스터링
├── market_aggregate.py           # [3] 비메뉴 집계
├── menu_aggregate.py             # [4] MENU 3계층 집계
├── golden_set_gen.py             #     검증 골든셋 생성
├── calc_cost.py / show_result.py #     비용 추정 / 결과 확인 헬퍼
└── archive/                      # v1 스크립트 보관

data/
├── ner_result_v2.json            # [1] 산출물 (specific 보존)
├── ner_result_v3.json            # [2] 정규화 산출물
├── market_item.csv               # [3] 비메뉴 character item ★
├── menu_item.csv                 # [4] MENU 평가 테이블 ★
├── norm_mapping/                 # 정규화·클러스터·분류 매핑
├── golden_set.csv                # 검증용
└── archive/                      # v1 데이터 산출물 보관
```

## 기술 스택

| 항목 | 내용 |
|---|---|
| NER / 정규화 / 분류 | gpt-4.1-mini (Batch API + 일반 API) |
| 구조화 출력 | json_schema / json_object |
| 가중치 | TF-IDF (수동 구현), 맛집 score (IDF 기반) |
