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
| **gguf** (파인튜닝·배포본, llama.cpp CPU) | **0.8766** | 0.8756 | **1.00** (3/3) | **2** | 5.98s | 무료(로컬) |
| **gpt-4.1-mini** (temp 0) | 0.7661 | 0.7694 | 0.33 (1/3) | 71 | 0.74s | $0.000293/건 ($0.114/389) |
| **base** Qwen2.5-1.5B-Instruct (원본·파인튜닝X) | 0.4267 | 0.4301 | 0.00 (0/3) | 60 | 5.61s | 무료(로컬) |

*(지연: 로컬은 CPU 20스레드 기준. gpt는 API 왕복. none Acc는 n=3이라 참고용.)*

## 핵심 해석

1. **파인튜닝 효과가 결정적**: 같은 1.5B인데 원본 **0.427 → 파인튜닝 0.877**로 약 2배. sLLM 파인튜닝이 이 태스크의 핵심.
2. **작은 로컬 파인튜닝 모델이 범용 GPT를 이김**: gguf **0.877 > gpt-4.1-mini 0.766**. "정상 경로는 로컬 sLLM"이라는 설계·비용 정당성의 직접 근거.
3. **출력 형식 준수(out_of_candidates)가 갈림**: base 60건·gpt 71건(≈15~18%)이 후보 title을 그대로 내지 않고 다른 말/변형을 출력 → 그만큼 오답. 파인튜닝 모델은 gguf 2건으로 **형식을 확실히 학습**. 룰 선택처럼 "정해진 레이블만 출력"해야 하는 태스크에서 파인튜닝의 큰 이점.
4. **none 판단**: 파인튜닝(gguf)만 3/3 정확. base는 "none"을 아예 못 냄(0/3), gpt도 1/3. (표본 3건이라 방향성만.)
5. **비용/지연 트레이드오프**: gpt는 건당 $0.0003·0.74s로 빠르지만 정확도·형식 준수가 낮음. gguf는 로컬 무료·정확하지만 CPU에서 ~6s(= M3 병목, GPU 전환 시 개선 지점).

## 한계

- 로컬 지연(5.6~6s)은 **CPU** 기준. GPU/vLLM이면 크게 단축(인터페이스 동일).
- none 표본 3건으로 none Acc는 참고용.
- gpt out_of_candidates 71건 중 상당수는 "후보에 가까운 말이지만 title 문자열 불일치"일 수 있음(완전일치 채점). 느슨한 매칭이면 gpt가 다소 오를 여지.
