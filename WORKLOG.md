# 개발 작업 기록 (WORKLOG) — git 이력 기반 전체 복원

> XAI 카페 추천 시스템. git 커밋 172개(PR 30개) 전수 기반으로 "실제로 한 일"을 시간순으로 복원.
> 기간: **2026-05-21 ~ 2026-07-22** (약 2개월). 작성: 2026-10-29.
> 모듈별 "왜/결과/예측" 회고는 `docs/retrospective.md`, 이 문서는 **시간순 작업 로그**.
> 커밋 해시는 `main` 기준. 날짜는 커밋 author date.

---

## 전체 타임라인 한눈에

| 단계 | 기간 | 주제 | 대표 PR |
|---|---|---|---|
| 0 | 05-21~05-26 | 프로젝트 셋업 + 리뷰 크롤링 | #2 |
| 1 | 05-27~06-02 | 전처리 v1 (ABSA/클러스터링) + 질문생성 | #3 |
| 2 | 06-02~06-06 | 임베딩 파인튜닝 v1 + 키워드선택 sLLM v1 | #3·#4·#5 |
| 3 | 06-06~06-12 | 질문 분류기 (RoBERTa→koelectra) | #6·#7 |
| 4 | 06-08~06-16 | Soar 룰 키워드 추출 + 룰 생성 | #8·#9·#10 |
| 5 | 06-16~06-17 | SQLite DB + 리뷰 요약 + 환경분리 | #12·#13 |
| 6 | 06-17~06-19 | LangGraph 파이프라인 1차 | #14~#18 |
| 7 | 06-22~06-24 | **대규모 v2 리빌드** (전처리/RAG/Soar/분류기/gguf) | #19~#26 |
| 8 | 06-26~06-27 | 서빙 구조 분리 + 컨테이너화 | #27 |
| 9 | 06-28~07-03 | 3-티어 룰 폴백 + README 정직화 | #28 |
| 10 | 06-30~07-05 | 평가셋(골드 qrels) + ML(LambdaMART) 비교 | #29·#30 |
| 11 | 07-22 | README 마감 | — |
| 後 | 10-28~29 | (정리 세션) 아카이브 분리 + 회고 문서화 | — |

---

## 단계 0 — 프로젝트 셋업 + 데이터 수집 (05-21~05-26, PR #2)

- `31b754f` Initial commit / `bfd5a9e` README
- `4873beb` Poetry 프로젝트 설정 → 이후 `9c2ac51`·`2733825`에서 **uv로 전환**(uv.lock 생성)
- `93c76f8` f-lab **자동 PR 리뷰 workflow** 추가(`.github/workflows/sonarcloud-analyze.yml`)
- `6d1b1cc` Python 3.10 + **네이버 리뷰 크롤링 완료** → 신림동 카페 **리뷰 15,233건** 수집
- `2df3e9c` 데이터 수집 폴더 구조화
- **산출물**: `data_collection/`, `shinline_cafe_reviews_test.csv`

## 단계 1 — 전처리 v1 (05-27~06-02, PR #3)

- `163cc9d` **ABSA 형식** 데이터 전처리 / `049051e` 질문 데이터 생성
- `dbd7876` 매장별 item 생성 초기 / `c224e17` 전처리 코드 수정
- `6ad5bcc` 키워드 **퍼지 매칭용 클러스터링** / `edbd285` 임베딩 모델 `bge-m3-ko`로 변경
- `712c017` **LLM으로 퍼지매칭 결과 평가·재군집** / `1c2409e`·`9f4967b` **2차 군집 클러스터링**
- `253be5f` 메뉴 키워드 정규화 완료(전 키워드 정규화)
- `e4b6df4` **리뷰 기반 market_item v1 완료**
- **포인트**: 이 시점 전처리는 v1. 이후 단계 7에서 2-pass NER로 **전면 재설계**됨.

## 단계 2 — 임베딩 파인튜닝 v1 + 키워드선택 sLLM v1 (06-02~06-06, PR #3·#4·#5)

- `96c114c` 임베딩 파인튜닝 **데이터셋 준비** / `9571aaf` **BGE-M3 파인튜닝 완료**(v1)
- `8d53f16` label별 train/test 분류 후 재파인튜닝 / `3150347` 설정 변경
- `2d3505f` 멀티 라벨 데이터 생성·정의
- `08949f2` 학습용 질문 생성 / `c87e0fa` keyword select 데이터 증강 / `25d8860` 질문·답변 필터링
- `8f90eef` **키워드 선택 모델 파인튜닝 코드 설계** / `3582571` **LoRA → QLoRA 전환**
- `d449004` **키워드 선택 모델(1.5B) 파인튜닝 완료**(v1)
- **산출물**: `embedding/`, `keyword_selection/`, HF 모델(v1)

## 단계 3 — 질문 분류기 (06-06~06-12, PR #6·#7)

- `e093b59` 유효성 검증 **학습 데이터 준비** / `7b57e65` 파인튜닝 코드 준비
- `1875aec` **RoBERTa 파인튜닝 성공**(klue/roberta)
- `e7eb656` 메뉴데이터 키워드 추출 수정 + validation 추가
- `a6c7983` **데이터 편향·오버피팅 문제** → 학습 데이터 재생성
- `16d7316` **invalid 질문 GPT로 신규 생성** (KLUE-MRC 뉴스 기반이 일상 invalid 못 잡아 전환)
- `7c03c78` 하드 질문 생성 / `a53ca30` **k-fold 교차검증** 도입(test 난이도 쉬움 문제)
- `da4b2b0` 하드 평가지표 검증 / `6272919` 폴더 리팩토링 / `e6ae86a` **하드 질문 재생성 후 파인튜닝 완료**
- **결과값** (hard test): roberta-base(aug후) Acc 0.80/F1 0.806 → **koelectra-base-v3 최종** Acc 0.80/F1 0.808, 추론 0.0066s. 5-fold Acc 0.9873±0.0053.
- **주의**: 이 모델은 단계 7에서 **서빙용으론 gpt-4.1-mini로 교체**됨(파인튜닝은 교체 근거 베이스라인으로 존속).

## 단계 4 — Soar 룰 키워드 추출 + 룰 생성 (06-08~06-16, PR #8·#9·#10)

- `09a15f3` **SOAR rule keyword extraction via OpenAI API**
- `f584866` sLLM 파인튜닝 데이터 재구축 / `8d905c6` test 데이터 생성·train/val 분리
- `74a3bd7` **Soar 룰 생성 코드 구축** + 예시 데이터 (`soar/Make_Rule/`)
- `0b243be` **OpenSearch에 타이틀별 키워드 업데이트** (Soar 룰용)
- **산출물**: `soar/`, `soar_rule_keywords.json`, `rule_metadata.json`
- **주의**: 정식 Soar 트랙. 단계 7에서 **Soar 엔진 제거 → Python trace**로 대체, 2026-10-29 `archive/soar/`로 이동.

## 단계 5 — SQLite DB + 리뷰 요약 + 환경 분리 (06-16~06-17, PR #12·#13)

- `1089c69` **SQLite DB 스키마 + 적재 스크립트**(`service/db/`)
- `ec186a5` 매장 item 키워드 기반 **리뷰 선별** / `59a22b4` 선별 리뷰 20→30개
- `ad16cb1` cafes 적재 기준을 **item 보유 매장**으로 수정
- `39621a3` **리뷰 기반 매장 정보 요약**(review_summary)
- `a0df00c` **Python 3.11 업그레이드 + 서비스/전처리 환경 분리**
- **산출물**: `data/cafe.db`, `data_collection/data_market_summary/`

## 단계 6 — LangGraph 파이프라인 1차 (06-17~06-19, PR #14~#18)

- `b0e016f` **LangGraph 기반 추천 파이프라인 1차 구현**
- `25adadb` **impasse 처리 노드** + 프롬프트 중앙화(`service/prompt.py`)
- `890dd9d` **Langfuse @observe 트레이싱** 연동 / `bf69471` Langfuse 4.x 호환 + impasse 시 OpenSearch tiebreak 업데이트
- `55b95f9` rule keywords를 JSON 대신 **OpenSearch에서 읽어** 최신 반영
- `85320c8` **최종 답변 스트리밍 노드**(`answer_generate`)
- `57177d8` Soar tiebreak **서브스테이트 깊이 off-by-one 버그 수정**
- `8d9cf2a` **menu-complex: 메뉴추출 + 룰선택 병렬 실행** 후 Soar
- `44d9f04` invalid/fallback 질문에 안내 메시지 응답
- **산출물**: `service/graph.py`, `service/nodes/*`

## 단계 7 — 대규모 v2 리빌드 (06-22~06-24, PR #19~#26) ★가장 밀도 높은 구간

### 7a. 전처리 v2 (PR #19 data-preprocessing-v2 · #20 data-normalization-v2)
- `89c88af` **2-pass NER 파이프라인**(배치) — entity/descriptor 분리(v1 "조용한 카페" 형용사 혼재 해결)
- `79775d6` 키워드 **정규화·개념 클러스터링 + 매장 item 집계**(TF-IDF)
- `8c3d6cd` NER 배치 **무한루프 방지**(반복 실패 청크 차단, 멱등)
- `7aacdec` README v2 + **v1 산출물·스크립트 archive 이동**
- **결과**: NER 201카페/entity 23,825(desc 94.4%), 클러스터링 TARGET 415→66·ATM 167→79·FAC 391→140. 산출물 `market_item.csv`·`menu_item.csv`.

### 7b. RAG/임베딩 v2 (PR #21 rag-system-v2)
- `f544cab` RAG용 rule 셋·임베딩 학습데이터 v2 재생성(비메뉴 293 rule)
- `935ce8a` multi rule 순열 중복 통합 + epoch 5 / `1c35ce6` **하드네거티브 학습** + 약한조합 제거
- `2866f27` **하이브리드(dense+sparse) 측정** → 이득 없음 / `5889390` **리랭커(bge-reranker) 측정** → 악화
- `a61d078` 메뉴+비메뉴 복합 증강 → `b45635f` **post-split으로 이동(train↔test 누수 제거)**
- `c7b7292` Recall@15/@20 추가 / `5e6dc6c` **신규데이터 기반 임베딩 재파인튜닝 완료**
- **결과값**: R@1 0.281→0.545, R@10 0.619→**0.888**, R@20 0.679→0.962. hybrid +0.5pt(기각), rerank 0.89→0.77(기각) → **dense 단독 확정**.

### 7c. 키워드선택 v2 (PR #22 keyword-selection-v2)
- `2895f6f` sLLM 파인튜닝 데이터 생성 완료(v2)
- `f9ab27a` **독립 조합 multi rule 재설계**(rule 중첩 해소) / `eb0d1df` README v2
- **결과값**: 전체 Acc **0.727 → 0.884** (+15.7%p). rule 180(동의어multi83)→177(single97+독립multi80).

### 7d. Soar v2 → Python trace (PR #23 soar-v2)
- `c139ffa` Soar 단계 v2 데이터 연동(177 rule operator/tiebreak + OpenSearch 저장)
- `6e2f1d7` **generic 키워드 descriptor 분리 매칭**(매장_넓다 vs 매장_깔끔하다)
- `1d5e4d4` **부정 키워드 negative tiebreak 도입**
- `78c31a7`·`da9889d` negative judge 페어비교 worst 선호(**무한 impasse 해결**)
- `0554deb`·`7ae97c0` negative를 Soar 룰 대신 **파이썬 후보 필터로**(no_output 해결)
- `a89621b`·`d0b8b77` Soar no_output 시 파이썬 trace 결과로 fallback
- `1da6fe2` **refactor: Soar 엔진 제거 — 규칙 기반 trace를 정식 추천 추론으로 (리뷰 반영)** ← 정식 Soar 포기 분기점

### 7e. 분류기 OpenAI 교체 + 메뉴처리 + gguf (PR #24 langgraph-v2 · #25 question-classifier-openai · #26 sllm-gguf-inference)
- `44bfb56` **OpenAI 분류기 hard_test(300) 평가 — koelectra 대비 교체 근거** (Acc 0.80→0.94)
- `a771f02` **질문 유형 분류 koelectra → OpenAI 교체**
- `32cc92c` **메뉴/메뉴+비메뉴 처리 구현**(맛집score 기반)
- `4520247` **sLLM rule 선택 gguf(q8_0)로 교체** — 로드 14.2s→1.1s, 추론 5.5s→1.9s

## 단계 8 — 서빙 구조 분리 + 컨테이너화 (06-26~06-27, PR #27 serving-fastapi)

- `af6d27a` **서빙 구조 분리 — FastAPI 상주 + 모델 추론 서버 분리 + 컨테이너화** (model:8001 / app:8000)
- `0ccd82c` **cafe.db 재현용 시드 커밋** / `ea9bbaf` `.env.example` + docker-compose + Quick Start
- `7fb2b07` **앱서버 컨테이너화 + 풀 docker-compose(CPU 배포 가능)**
- `d8967dd` 인덱싱 **2단계 부트스트랩 명시**(uodate_index 누락 → 추천 안 나오던 문제)
- `339a13c` uv.lock 재생성(웹 머지로 깨진 lock 해소)
- **산출물**: `inference/`, `service/Dockerfile`, `docker-compose.yml`, `.env.example`

## 단계 9 — 3-티어 룰 폴백 + README 정직화 (06-28~07-03, PR #28 rule-fallback)

- `3b422fb` **rule=none LLM 폴백 노드** 추가(복합·모호 질문도 추천)
- `a88753d` impasse_resolve 모델 gpt-4.1 → **gpt-4.1-mini**(지연 단축)
- `fe7e8e7` **신규 룰 생성 티어(rule_create) + OpenSearch 저장** — CogRec식 온라인 학습
- `be39494` **README 정직화** — track1은 Soar 영감 규칙기반 trace(정식 Soar 아님)
- **결과**: 티어0(sLLM) → 티어1(LLM 근접) → 티어2(LLM 신규생성) 3단 폴백 완성.

## 단계 10 — 평가셋 + ML 비교 (06-30~07-05, PR #29 ml-ranking · #30 recsys-eval)

- `b28b0e5` **LambdaMART 랭킹 POC 노트북** + LLM-judge 평가 산출물
- `17b736c` **골드 qrels 평가셋** + 룰 vs ML(LambdaMART) 비교 하니스
- **결과값**(gold qrels, 50질문, rel≥2): **rule hit@1 0.60 / ndcg@3 0.45**, **ml hit@1 0.84 / ndcg@3 0.80** → ML 우세 실증.
- ※ 이후 2026-10-29 방향 재검토에서 **"LambdaMART 같은 ML은 쓰지 않는다"** 결정 → ML 트랙 `archive/ml/`로 이동. 최종 deliverable = LLM 서비스.

## 단계 11 — 마감 (07-22)

- `7ea90b2` README 수정

---

## 後기 — 정리 세션 (2026-10-28~29, git 미추적 docs 작업 포함)

- LLM 서비스/패키징 전수 파악, 사용·미사용 코드 감사.
- 문서 드리프트 수정: 서빙 분류기 `koelectra → gpt-4.1-mini` (service/README·CLAUDE.md).
- **아카이브 분리**(삭제 아님): `archive/soar/`(sml 148MB 포함)·`archive/ml/`·`archive/misc_unused/`. 라이브 서비스 무결성 확인, `data/soar_rule_keywords.json`은 서비스 사용분이라 존치.
- 회고 문서 `docs/retrospective.md`(모듈별 왜/결과/[예측]) + 본 `WORKLOG.md` 작성.

---

## 핵심 성과 수치 모음 (근거: 각 모듈 README / eval 산출물)

| 항목 | 수치 |
|---|---|
| 수집 리뷰 | 15,233건 |
| NER | 201카페 / entity 23,825 / descriptor 94.4% |
| 임베딩 R@10 | 0.619 → **0.888** (파인튜닝) |
| 키워드선택 Acc | 0.727 → **0.884** (독립 multi 재설계) |
| sLLM gguf 추론 | 5.5s → **1.9s** |
| 질문분류 교체 | koelectra 0.80 → gpt-4.1-mini **0.94** (hard F1) |
| end-to-end | rule hit@1 0.60 / ml 0.84 |

> 상세 설계 의도·기각 사유·예측은 `docs/retrospective.md` 참조.
