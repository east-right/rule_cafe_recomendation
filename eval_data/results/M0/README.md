# M0 — 수정 전 기준선 (baseline)

- **커밋**: `48c67e1` (branch `fix/sentiment-matching`)
- **측정일**: 2026-11-02
- **입력**: `eval_data/data/questions_eval.json` 50문항
- **러너**: `eval_data/run_service_eval.py` (실제 서비스 LangGraph를 그대로 실행)
- **산출물**: `baseline.json`(권위본·50문항), `b3_evidence.json`(B3 증거), `baseline_dry2.json`·`baseline_dry10.json`(스모크/dry 중간물)

## 측정 조건 (중요)

- **실제 서비스 경로** 실측이다. `graph.stream()`으로 question_valid→rule_select/menu_extract→soar_recommend→(impasse)→answer_generate 를 그대로 태웠다. `eval_pipeline.py`의 점수기반 top-3 와는 **다른 경로**다(그건 오프라인 랭킹 근사).
- **impasse 영구 저장을 하네스에서 차단한 상태로 측정**했다. 서비스 코드는 수정하지 않고, 측정 프로세스에서만 두 영구쓰기를 no-op 으로 패치:
  - `impasse_resolve._update_os_tiebreak` (B3 미수정 동작: tiebreak 키워드 무검증 영구 append)
  - `rule_create._persist_rule` (Tier-2 신규 생성룰 persist)
  - → 문항별 추론·추천은 in-memory 로 동일하게 수행되고, **OpenSearch 인덱스 오염/문항순서 의존만 제거**해 재현성을 확보했다. (`b3_evidence.json` 참조)
- **Langfuse 트레이싱 OFF**(`LANGFUSE_TRACING_ENABLED=false`) — M3 지연 노이즈 제거.
- **답변 생성은 temperature=0.7** 라 답변 텍스트는 재실행마다 달라진다(룰 경로는 거의 결정적). 이 baseline 의 **답변은 B2 수정 전 스냅샷**으로 M6(설명 일치도)에서 재사용한다.

## 결과 요약 (50문항, 오류 0)

| 질문 유형 | 수 |
|---|---|
| non-menu | 37 |
| menu-only | 3 |
| menu-complex | 3 |
| fallback(invalid/비한국어) | 7 |

- **룰 선택 성공(실제 룰)**: 38 / none·fallback 경로: 12
- **soar 결과**: 단일 추천 37 · no_output(operator 0매칭) 2 · 최종 impasse 1 · soar 미실행(fallback·menu-only 캔드) 10
- **최종 단일 추천 매장 확보**: 37문항
- **Rule Fallback(Tier-1) 사용**: 9문항 · **Tier-2 rule_create 발동**: 0
- **impasse 발동**: 11문항 (q074 최다 5회). 미수정 프로덕션이었다면 이만큼 무검증 영구쓰기 발생 → `b3_evidence.json`

### M2 미리보기 (요청당 비용 — 상세는 M2에서)

- 유효 run OpenAI **121콜** / in **186,053 tok** / out **4,230 tok**
  - gpt-4.1-mini 82콜 (in 167,506 · out 313, usage 정확)
  - gpt-4.1 39콜 (in 18,547 · out 3,917, tiktoken 추정)
- **핵심 발견**: `menu_extract` 1콜이 input ~22K 토큰 — 전체 메뉴 리스트를 매번 통째로 프롬프트에 넣는다. 입력 토큰의 대부분을 차지. (M2 비용 최적화 1순위 후보)
- 비용 ≈ **$0.14** (gpt-4.1 $2/$8·mini $0.4/$1.6 per 1M 공개단가 기준, 요율 확인 필요·소액).
- 참고: 1차 run 은 모델서버 OOM 으로 실패(부분 ~58콜 소모) 후 재기동·재실행. M0 총 지출 ≈ $0.20.

### M3 미리보기 (지연 — 상세는 M3에서)

- end-to-end: mean **5.02s** · p50 **5.6s** · p95 **7.96s**
- 답변 첫 토큰까지: mean **0.44s**
- 병목: rule_select(모델서버 /embed + GGUF /select_rule) + answer(gpt-4.1 스트리밍). 노드별 평균은 `baseline.json > summary.latency.by_node_mean_s`.

## 함께 관측된 버그 증거 (이번 단계에서 수정 안 함)

- **B4 (menu-complex 폴백 우회)**: `q052` menu-complex + 룰선택 none → Rule Fallback 우회 → **추천 실패(cafe=None)**. `q104`도 none 이지만 메뉴 경로로 커피니쉬 추천됨. `q139`는 룰(공부) 선택돼 정상. → B4 수정의 근거.
- **B3**: `b3_evidence.json` (dry10 실측 2건 + full M0 차단 11건).

## 재현

```bash
# 선행: OpenSearch(:9200, cafe_rules 177) + 모델서버(inference.server :8001)
uv run --with tiktoken python eval_data/run_service_eval.py          # 전체 50
uv run --with tiktoken python eval_data/run_service_eval.py --dry 10 # dry
```
룰 경로(classify·rule_select·soar)는 temperature=0 이라 재실행 시 선택 룰·추천 매장 동일. 답변 텍스트만 변동.
