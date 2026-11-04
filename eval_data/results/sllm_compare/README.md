# Rule 선택 정확도 — 모델별 비교 (389문항)

- **데이터**: `keyword_selection/data/test_with_candidates.jsonl` (389문항, 각 후보 15개, 정답은 389/389 모두 후보 내 존재, none 3건)
- **프롬프트**: `keyword_selection/prompt.py` `SYSTEM_PROMPT` + `build_user_prompt` (전 모델 동일, 모델만 교체)
- **디코딩**: 로컬 greedy(do_sample=False, repetition_penalty 1.1) / gpt temperature 0 / max_new_tokens 32
- **러너**: `keyword_selection/compare_sllm.py` (모델당 별도 프로세스 = OOM 방지). 커밋은 `results.json` 참조.
- 정답 판정: `pred == answer` 완전일치. 출력이 후보 title(∪"none") 밖이면 **out_of_candidates**로 따로 집계(+오답 처리).

> **adapter(파인튜닝 HF LoRA)는 gguf와 동일 파인튜닝의 다른 포맷**(gguf=q8_0 양자화 배포본)이라 중복 → 비교에서 제외. 배포 실체인 **gguf**로 대표.

## 결과 (실측)

| 모델 | 전체 Acc | match Acc | none Acc (n=3) | out_of_cand | 건당 지연 | 건당 비용 |
|---|---|---|---|---|---|---|
| **gpt-4.1-mini (Structured Output, enum 강제)** | **0.9306** | 0.9352 | 0.33 (1/3) | **0** | 0.72s | $0.000386/건 ($0.150/389) |
| **gguf** (파인튜닝·배포본, llama.cpp CPU) | **0.8766** | 0.8756 | **1.00** (3/3) | **2** | 5.98s | 무료(로컬) |
| **gpt-4.1-mini** (일반, temp 0) | 0.7661 | 0.7694 | 0.33 (1/3) | 71 | 0.74s | $0.000293/건 ($0.114/389) |
| **base** Qwen2.5-1.5B-Instruct (원본·파인튜닝X) | 0.4267 | 0.4301 | 0.00 (0/3) | 60 | 5.61s | 무료(로컬) |

*(실행 런타임: **base=HF transformers fp32 CPU**, **gguf=llama.cpp CPU**(서로 다른 런타임), gpt=API. 로컬 지연은 CPU 20스레드. none Acc는 n=3이라 참고용.)*

## 핵심 해석

1. **파인튜닝 효과가 결정적**: 같은 1.5B인데 원본 **0.427 → 파인튜닝 0.877**로 약 2배. sLLM 파인튜닝이 이 태스크의 핵심.
2. **출력 형식 준수(out_of_candidates)가 점수를 좌우**: 일반 gpt 71건·base 60건(≈15~18%)이 후보 title을 그대로 안 내고 다른 말/변형을 출력 → 그만큼 오답. 파인튜닝(gguf 2건)·Structured Output(gpt_struct 0건)은 형식을 강제/학습해 이 손실이 없음.
3. **Structured Output이 gpt의 진짜 실력을 드러냄**: enum 강제 시 gpt-4.1-mini **0.766 → 0.931**(ooc 71→0). 즉 **일반 gpt의 낮은 점수는 추론이 아니라 출력 형식 문제**였다. 공정(형식 보장) 비교에선 **gpt_struct 0.931 > gguf 0.877** — 추론 자체는 GPT가 더 정확.
4. **그럼에도 gguf의 포지션**: 로컬·무료·형식 강제 없이도 0.877로 일반 gpt를 앞섬. "정상 경로는 로컬 sLLM"의 근거는 **정확도 우위가 아니라 비용·온프레미스·형식 robustness**로 재정의되어야 함(정직한 수정). 최고 정확도가 목표면 gpt+Structured Output이 더 낫다.
5. **none 판단은 형식이 아니라 추론**: enum에 "none"이 있어도 gpt_struct는 여전히 1/3(형식 아닌 판단 문제). 파인튜닝 gguf만 3/3. (표본 3건, 방향성.)
6. **비용/지연**: gpt_struct $0.00039·0.72s(API). gguf 로컬 무료지만 CPU ~6s. 정확도 최우선이면 gpt_struct, 비용/온프레미스 최우선이면 gguf.

## 지연 분해 — "문서 1.9s vs 이번 5.98s"

GGUF(llama.cpp CPU) 지연을 후보 수별로 각 20건 재측정(워밍업 1회 후):

| 후보 수 | 평균 지연 | min~max |
|---|---|---|
| 10개 (서비스 TOP_K 조건) | **3.47s** | 0.23~3.93 |
| 15개 (이번 평가 조건) | **5.66s** | 4.85~6.07 |

- **후보 수(=프롬프트 길이)가 지연의 주요인**: 10→15개로 늘자 +2.2s. 이번 평가가 5.98s인 건 **15후보**라서. 서비스(10후보)면 ~3.5s로 떨어짐.
- 다만 측정값(10후보 3.47s)도 문서 **1.9s보다 큼**. 프로브는 고정 질문("질문")을 써서 llama.cpp 프롬프트 프리픽스 캐시가 일부 작동(그래서 min 0.23 등 분산 큼) → 그럼에도 평균이 1.9s보다 높음. 문서 1.9s는 **다른 조건**(더 짧은 후보·디코딩만 측정·다른 머신·웜 상태)으로 추정되며, 동일 조건 재현 없이는 정합 불가 → 문서 수치를 **측정 조건과 함께 갱신 필요**.
- 참고: 위 base(5.61s)는 **HF transformers fp32**, gguf(5.98s)는 **llama.cpp** — 런타임이 달라 두 로컬 수치의 직접 비교는 부적절.

## 한계

- 로컬 지연(5.6~6s)은 **CPU** 기준. GPU/vLLM이면 크게 단축(인터페이스 동일).
- none 표본 3건으로 none Acc는 참고용.
- gpt out_of_candidates 71건 중 상당수는 "후보에 가까운 말이지만 title 문자열 불일치"일 수 있음(완전일치 채점). 느슨한 매칭이면 gpt가 다소 오를 여지.
