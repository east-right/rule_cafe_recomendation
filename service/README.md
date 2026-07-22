# service/ — 추천 서비스 파이프라인 (LangGraph 에이전트)

질문을 받아 카페를 추천하고 근거와 함께 답변을 스트리밍하는 **실서비스 런타임**.
LangGraph로 노드를 잇고, 각 단계는 Langfuse로 추적한다.

> 이 모듈은 "룰 선택 → 룰기반(soar 영감) 추천 → 설명 답변" 흐름을 구현한다.
> 모델(임베딩·sLLM)은 `inference/` 서버가, 룰 검색은 OpenSearch가 담당하고
> service는 이들을 오케스트레이션한다.

## 실행

```bash
# 선행: OpenSearch(9200) + 모델 서버(inference.server, 8001) 기동, cafe.db·cafe_rules 인덱스 준비
uv run uvicorn service.api:app --host 0.0.0.0 --port 8000   # 상주 API
# 또는 CLI 단발
uv run python -m service.main "조용하고 넓은 카페 추천해줘"
```

- `POST /recommend {"question": "..."}` → 답변 토큰 스트리밍
- 진입점: `api.py`(HTTP) / `main.py`(그래프 invoke + `answer_generate` 스트리밍)

## 파이프라인 흐름 (`graph.py`)

```
question_valid  (koelectra 분류 + 한국어 게이트)
  ├ non-menu ─────────────▶ rule_select
  ├ menu-only ────────────▶ menu_extract ──▶ END (맛집 2곳 바로 답변)
  ├ menu-complex ─────────▶ [menu_extract ∥ rule_select] (병렬 join)
  └ invalid/fallback ─────▶ END (답변 불가 안내)

rule_select  (BGE-M3 검색 → GGUF sLLM 룰 선택)
  ├ 룰 매칭 ──────────────▶ soar_recommend
  └ none ─────────────────▶ rule_fallback

rule_fallback  (LLM이 후보 top-10 중 가장 가까운 1~3룰 강제 선택)
  ├ 룰 선택 ──────────────▶ soar_recommend
  ├ NONE_FIT ─────────────▶ rule_create
  └ 실패 ─────────────────▶ END

rule_create  (LLM이 어휘 내 신규 룰 생성 + OpenSearch 영구 저장)
  ├ 생성 성공 ────────────▶ soar_recommend
  └ 실패 ─────────────────▶ END (억지 추천 금지)

soar_recommend  (트레이스 필터링: operator → negative → tiebreak)
  ├ 1곳 ──────────────────▶ END → 답변
  ├ 2곳+ (impasse) ───────▶ impasse_resolve
  └ 0곳 ──────────────────▶ END (추천 없음)

impasse_resolve  (LLM이 남은 후보 가를 tiebreak 키워드 추가, OS 갱신)
  └──────────────────────▶ soar_recommend  (재귀, 최대 5회)
```

## 3-티어 룰 폴백 (XAI 핵심)

질문이 어려워도 죽지 않고, **점점 더 비싼 수단**으로 추천을 이어간다:

| 티어 | 노드 | 방식 | 비용 |
|---|---|---|---|
| 0 | `rule_select` | sLLM(GGUF)이 후보 중 **정확히** 맞는 룰 선택 | 싸다(로컬) |
| 1 | `rule_fallback` | 정확 매칭 없음(none) → LLM이 후보 중 **가장 가까운** 1~3룰 선택 | 중간 |
| 2 | `rule_create` | 후보 전부 무관(NONE_FIT) → LLM이 **신규 룰 생성** 후 OpenSearch 저장 | 비쌈 + 온라인 학습 |

- **환각 방지**: 폴백/생성 모두 키워드를 지어내지 않고 **실제 어휘(OpenSearch 룰 정의 / `cafe_keywords`)** 안에서만 고른다. operator를 못 만들면 정직하게 "추천 없음".
- **온라인 학습**: `rule_create`가 만든 룰은 `cafe_rules`에 영구 인덱싱 → 다음에 비슷한 질문이 오면 RAG가 바로 후보로 찾음.

## 노드별 역할 (`nodes/`)

| 파일 | 역할 |
|---|---|
| `question_valid.py` | koelectra 분류(non-menu/menu-only/menu-complex/invalid) + 한국어 비율 게이트 |
| `menu_extract.py` | 질문에서 메뉴 추출(LLM). menu-only=맛집 상위 2곳 / menu-complex=해당 메뉴 매장 전부 |
| `rule_select.py` | BGE-M3 임베딩(모델서버) → OpenSearch KNN top-10 → sLLM 룰 선택 + 키워드 조회 |
| `rule_fallback.py` | 티어1. 후보 중 근접 룰 1~3개, 2순위 operator를 tiebreak로 흡수 |
| `rule_create.py` | 티어2. 어휘 내 신규 룰 생성 + `cafe_rules` 영구 저장 |
| `soar_recommend.py` | 룰기반 추론. operator(후보)→negative(narrowing)→tiebreak(우열)로 좁히고 각 단계 trace 기록 |
| `impasse_resolve.py` | 교착(2곳+) 시 LLM이 구분 tiebreak 키워드 1개 추가, OS 업데이트(비동기). 최대 5회 |
| `answer_generate.py` | 최종 답변 스트리밍(gpt-4.1). trace의 유효 키워드를 근거로 사용. *그래프 밖, `main.stream`이 호출* |

## 상태 (`state.py`)

`AgentState`(TypedDict): 노드 간 전달되는 공유 상태.
`question` / `question_type` / `rule_candidates` / `selected_rule` /
`operator·tiebreak·negative_keywords` / `soar_result` / `impasse_iterations` /
`extracted_menu` / `menu_cafe_names` / `rule_fallback_used` / `needs_rule_creation` / `rule_created` 등.

## DB (`db/`)

SQLite `data/cafe.db` — 서비스가 읽는 매장 데이터.

| 파일 | 역할 |
|---|---|
| `schema.py` | 테이블 정의: `cafes` · `cafe_keywords` · `menus` · `cafe_menus` (+ 인덱스). `init_db()` |
| `loader.py` | CSV(`market_item`·`menu_item`·`generic_keywords`) → DB 적재. `--full`은 전체 재적재 |
| `queries.py` | 공통 쿼리 (operator 키워드 매칭 매장 조회 등) |

```bash
uv run python service/db/loader.py          # cafe_keywords 재적재
uv run python service/db/loader.py --full   # cafes 포함 전체 재적재
```

## 프롬프트 (`prompt.py`)

각 LLM 노드의 system/user 프롬프트 모음 (MENU_EXTRACT / RULE_FALLBACK / RULE_CREATE / IMPASSE / ANSWER).

## 외부 의존성

- **OpenSearch** (`cafe_rules` 인덱스): 룰 검색·키워드 조회·신규 룰 저장
- **모델 서버** (`inference.server`, :8001): BGE-M3 임베딩 + GGUF sLLM 룰 선택
- **OpenAI API**: 메뉴추출·폴백·룰생성·impasse·최종답변 (gpt-4.1 / gpt-4.1-mini)
- **Langfuse**: 노드별 트레이싱

> 참고: 최근 프로젝트 방향은 룰/Soar 기반 → **학습 랭커(LambdaMART) 기반**으로 전환 중.
> 이 service 모듈은 룰기반 트랙의 서빙 구현이며, ML 트랙 평가·비교는 `eval_data/` 참조.
