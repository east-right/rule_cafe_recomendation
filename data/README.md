# data/ — 데이터 카탈로그

모든 모듈이 참조하는 공용 데이터 폴더. 파일이 파이프라인 단계별로 평평하게 쌓여 있어
(스크립트들이 `data/xxx` 경로를 하드코딩하므로 **파일명·위치는 함부로 바꾸지 말 것**).

★ = 핵심(다른 단계가 계속 참조) · ☆ = 중간 산출물 · (옛) = 지난 접근/보관

## 데이터 흐름 한눈에

```
[크롤링]  shinline_cafe_ids_v2 → shinline_cafe_reviews_test ─┐
                                                             ▼
[NER]     preprocesing_batch → ner_result_v2 ──normalize──▶ ner_result_v3
                                     │                          │
                                     ▼ (norm_mapping)           ▼ (norm_mapping)
[집계]                          menu_item.csv            market_item.csv ─┐
                                     │                          │         │
[요약]                               └──▶ market_review_summary ◀┘         │
                                                                          ▼
[룰]      market_item ──클러스터/병합──▶ rule_metadata → rule_metadata_merged ★
                                                    │
                                                    ├─▶ soar_rule_keywords ★ (operator/tiebreak/negative)
                                                    └─▶ finetune_embd_query_pos (BGE-M3 학습용)
[질문]    market_item ──GPT──▶ valid_questions ★ → (classifier 학습 / ml POC)
[서빙]    loader.py: market_item + menu_item + generic_keywords ──▶ cafe.db ★
```

## 1. 원천 데이터 (크롤링)

| 파일 | 설명 |
|---|---|
| `seoul_restaurants_shinline.csv` | 크롤링 원본 — 신림동 카페/식당 목록 |
| `shinline_cafe_ids_v2.csv` | 카페 place_id 목록 |
| ★ `shinline_cafe_reviews_test.csv` | **원본 리뷰 전체** (모든 전처리의 시작점) |
| `reviews_clean.csv` | 위 리뷰에서 완전중복 제거 (`eval_data/prep_reviews.py`, 평가용) |

## 2. NER 전처리

| 파일 | 설명 |
|---|---|
| `ner_result_v2.json` (10.8MB) | 리뷰 → LLM NER 추출 원본 (`data_preprocessing/preprocesing_batch_while.py`) |
| `ner_result_v3.json` (7.4MB) | v2 정규화본 (`normalize.py`, `norm_mapping/` 사용) |
| `norm_mapping/` | 정규화 매핑 (cluster/desc/entity/vocab × ATMOSPHERE·FACILITY·TARGET·MENU) |

## 3. item 집계

| 파일 | 설명 |
|---|---|
| ★ `market_item.csv` | 비메뉴(FACILITY/ATMOSPHERE/TARGET) 매장별 item + TF-IDF (`market_aggregate.py`). **룰·질문·DB의 공통 입력** |
| `menu_item.csv` | 메뉴 3계층 집계 (`menu_aggregate.py`) |
| `generic_keywords.json` | 범용 키워드 (DB 적재 시 operator 매칭용, `service/db/loader.py`) |

## 4. 리뷰 요약

| 파일 | 설명 |
|---|---|
| `market_review_summary.json` | 매장별 리뷰 요약 (`data_collection/data_market_summary/`) → `cafe.db`의 review_summary |
| `review_summary_batch_output.jsonl` | 요약 배치 산출물 *(gitignore, 로컬 전용)* |

## 5. 룰 메타데이터 (★ 핵심 자산)

| 파일 | 설명 |
|---|---|
| `rule_metadata.json` | 룰 메타 원본 |
| ★★ `rule_metadata_merged.json` | **177개 병합 룰** (title+description+keywords). embedding 인덱싱 · keyword_selection 학습 · eval 검색의 공통 소스 |
| ★★ `soar_rule_keywords.json` | 룰별 operator/tiebreak/negative 키워드. soar 추천 · eval 룰기반의 핵심 |
| `title_groups.json` ☆ | 룰 title 병합 중간 산출 |

## 6. 임베딩 파인튜닝

| 파일 | 설명 |
|---|---|
| `finetune_embd_query_pos.json` | BGE-M3 파인튜닝용 query-positive 쌍 (`data_collection/.../build_embd_data.py`) |

## 7. 질문 데이터

| 파일 | 설명 |
|---|---|
| `cafe_questions.json` | 초기 생성 질문 (`data_collection/question_arg.py`) |
| ★ `valid_questions.json` | **유효 질문 + 키워드 라벨** (`generate_questions.py`). question_classifier 학습 · ML POC 입력 |
| `golden_set.csv` | question_classifier 골든셋 (`data_preprocessing/golden_set_gen.py`) |

## 8. ML 랭킹 POC — 옛 평가셋 (키워드 기반)

> ⚠️ **`eval_data/data/qrels.csv`(새 골드)와 다름.** 이건 `notebooks/ml_ranking_poc.ipynb`가 쓰는
> 옛 키워드-질문 기반 qrels. 새 방법론(전매장×전리뷰 홀리스틱 LLM-judge)이 이걸 대체함.

| 파일 | 설명 |
|---|---|
| `qrels.csv` (옛) | 키워드질문 + 풀링 기반 관련도 라벨 |
| `query_set.json` (옛) | POC 질문셋 |
| `pools.json` (옛) | 질문별 후보 카페 풀 |

## 9. 서비스 DB

| 파일 | 설명 |
|---|---|
| ★★ `cafe.db` | SQLite 시드 (cafes · cafe_keywords · menus · cafe_menus). service·ml·eval 공통. `.gitignore`에서 `*.db` 제외 대상으로 커밋됨 |

## archive/

지난 버전 보관 (`*_v1.json`, `rule_metadata_merged_premulti*` 등). 현행 아님, 참고용.

---

## 핵심만 추리면

실제로 계속 살아 움직이는 **핵심 6개**:
- `cafe.db` — 서비스/평가의 DB
- `rule_metadata_merged.json` — 룰 정의(177개)
- `soar_rule_keywords.json` — 룰 키워드(operator/tiebreak/negative)
- `market_item.csv` — 매장 item 집계(모든 하류의 입력)
- `valid_questions.json` — 질문+키워드
- `shinline_cafe_reviews_test.csv` — 원본 리뷰

나머지는 이 6개를 만들기 위한 **중간 산출물**이거나 **지난 접근(옛 qrels/archive)**이야.
