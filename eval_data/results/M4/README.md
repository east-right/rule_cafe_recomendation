# M4 — 예외 경로 동작 (3-Tier 폴백 · Impasse · none)

- **커밋**: `b6b51e3` (현 B2~B4 코드) · **평가셋**: `qrels_181.csv` · **실행**: full 50 (`run.json`, 실측 125콜)
- **Tier**: Tier0=1차 sLLM `rule_select` 성공, Tier1=`rule_fallback`, Tier2=`rule_create`. menu-only/invalid는 룰tier 아님.

## Tier 발동 (50문항)

| 경로 | 문항수 | 최종 추천 Hit@1 |
|---|---|---|
| Tier0 (정상 sLLM) | 29 | **0.8571** (28곳 추천) |
| Tier1 (Rule Fallback) | 11 | **0.7273** (11곳 추천) |
| Tier2 (rule_create) | **0** | — (미발동) |
| menu-only (메뉴 2곳) | 3 | (별도 경로) |
| invalid (캔드 응답) | 7 | — |

## Impasse · 최종 none

| 지표 | M4 (B2~B4) | M0 (수정 전) |
|---|---|---|
| impasse 발동 문항 | 12 (총 17회 반복) | 11 |
| **rule경로 최종 추천 실패(none)** | **1곳** (q150) | **3곳** (q052·q074·q150) |

## 핵심

- **추천 실패 3 → 1로 감소**: B4가 q052(menu-complex none→홍카페), B3가 q074(impasse 소진→파스쿠찌) 해결. 남은 q150(깊은밤)은 operator 매칭 0(no_output)으로, B2~B4 범위 밖(룰 어휘 커버리지 문제).
- **Tier0 Hit@1 0.857 > Tier1 0.727**: 1차 sLLM이 정확히 고른 경우가 가장 정확. 폴백은 커버리지를 11문항 회복하되 품질은 소폭 낮음 — "폴백은 안전망, 정상 경로가 최선"이라는 설계 의도와 일치.
- **Tier2(신규 룰 생성) 0회**: 평가 50문항에선 기존 177룰 + 폴백으로 충분, NONE_FIT 미발생. (Tier2는 더 희귀한 질문에서만 발동 — 측정셋 한계)
- impasse는 B3로 매 반복이 반드시 narrowing → 12문항 중 11곳이 단일 추천으로 해소(q150만 no_output).

## 한계

- menu-only 3곳의 Hit@1은 본 집계에 미포함(soar 단일 cafe가 아니라 메뉴 상위 2곳 추천). 별도 평가 필요.
- Tier2 미발동은 평가셋이 기존 룰로 대부분 커버되기 때문 — 커버리지 밖 질문셋으로 별도 검증 권장.
