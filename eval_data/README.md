# eval_data — 골드 qrels 생성 파이프라인

신림동 카페 추천 시스템의 **모델-독립적 평가셋(gold qrels)** 을 만드는 파이프라인.
LLM이 실제 리뷰를 읽고 (질문, 매장) 쌍을 0–3점으로 채점 → NDCG, MRR 등 랭킹 메트릭으로 규칙 기반·LambdaRank 공정 비교.

## 핵심 설계 원칙

- **키워드 없음**: 키워드 기반 레이블 → 키워드 기반 채점 = 순환 오류. LLM이 원본 리뷰 텍스트만 보고 판단.
- **존재가 아닌 대표성**: "공부 리뷰 1개 있음" ≠ 공부 카페. 그 요구가 카페의 실제 강점/정체성인지 본다.
- **특정성**: "딸기 케이크" 질문엔 딸기 케이크 직접 증거 필요. 치즈케이크·마카롱으로 대체 불가.

## 데이터 파일 (`data/`)

| 파일 | 설명 |
|------|------|
| `questions.json` | 전체 153개 질문 (8개 테마, zero-shot 생성 + 중복 제거) |
| `questions_eval.json` | 평가용 50개 (테마 비례 층화 추출, seed=0) |
| `review_selection.json` | (qid, 매장)별 전송할 리뷰 인덱스 (코사인 유사도 top-50 캡) |
| `cafes.json` | 매장 목록 205개 (정렬 순서 보존용) |
| `qrels.csv` | **최종 산출물** — 10,250행 (50질문 × 205매장) |

### qrels.csv 컬럼

| 컬럼 | 설명 |
|------|------|
| `qid` | 질문 ID (q000–q049) |
| `cafe` | 매장명 |
| `grade` | 관련도 0–3 |
| `is_strength` | 이 요구가 카페의 실제 강점인지 (true/false) |
| `reason` | 채점 근거 한 줄 |
| `evidence` | 근거 리뷰 인용 (최대 3개, `\|` 구분) |

### grade 기준

| 점수 | 의미 |
|------|------|
| 3 | 명백한 대표 강점 — 반복·중심 증거, 사람들이 그 목적으로 찾는 곳 |
| 2 | 분명한 강점이나 대표까지는 아님 |
| 1 | 가능하나 부수적·스침. 카페 정체성은 다른 데 있음 |
| 0 | 무관하거나 반하는 근거 우세 |

## 스크립트 구조

```
eval_data/
├── prep_reviews.py       # 리뷰 중복 제거 → data/reviews_clean.csv (루트 data/)
├── gen_questions.py      # 질문 생성 (OpenAI Batch) → data/questions.json
├── select_questions.py   # 층화 추출 50개 → data/questions_eval.json
├── select_reviews.py     # 유사도 top-50 선택 → data/review_selection.json
├── judge.py              # LLM 채점 로직 + 루브릭
├── run_judging.py        # 배치 오케스트레이션 → data/qrels.csv
└── batch_runner.py       # OpenAI Batch API 재사용 하니스
```

## 실행 순서

> 이미 `data/qrels.csv`가 있으면 아래 1–4는 불필요.

```bash
# 0. 리뷰 중복 제거 (루트에서)
uv run python eval_data/prep_reviews.py

# 1. 질문 생성 (배치, ~10분)
uv run python eval_data/gen_questions.py

# 2. 50개 추출
uv run python eval_data/select_questions.py

# 3. 유사도 리뷰 선택 (임베딩 API 호출, ~5분)
uv run python eval_data/select_reviews.py

# 4. 전수 채점 (배치 5개, 50 × 205 = 10,250건, gpt-4.1-mini, ~$3)
uv run python eval_data/run_judging.py submit
uv run python eval_data/run_judging.py fetch   # 완료 후 실행

# 배치 일부 실패 시
uv run python eval_data/run_judging.py retry
```

## 채점 규모

- 질문 50 × 매장 205 = **10,250 요청**
- 매장당 리뷰 최대 50개 (코사인 유사도 top-50)
- 모델: `gpt-4.1-mini`, temperature=0
- 배치 5개 (CHUNK=2050, 50% 할인 적용)
- 비용: 약 $3–4

## 다음 단계

`qrels.csv`를 기준으로 랭킹 시스템 비교:

```python
# 예시
from eval import ndcg_at_k, mrr, hit_at_k, recall_at_k
scores = evaluate(system_ranked_output, qrels)
```

비교 대상: 규칙 기반(Soar trace) vs LambdaRank vs BM25 baseline
