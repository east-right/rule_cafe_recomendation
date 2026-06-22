# SOAR Rule Generation

## 개요
`soar_rule_keywords.json`에서 추출된 keyword를 기반으로 SOAR 추론 로직을 `.soar` 파일로 시각화합니다.

이 모듈은 두 가지 용도로 활용됩니다.

1. **문서화 및 검증**: 생성된 rule의 논리 구조를 `.soar` 포맷으로 시각화하여 검토
2. **메인 시스템 연동**: `generate_soar_rule(rule)` 함수를 그대로 사용하여 실시간 신규 rule 생성

> **참고**: 이 프로젝트는 실제 SOAR 런타임 엔진(JSoar, CSoar)을 사용하지 않습니다.
> `.soar` 파일은 SOAR 아키텍처의 추론 논리를 문서화하고 시각적으로 검증하기 위한 용도입니다.
> 실제 추론은 이 구조를 기반으로 Python으로 구현됩니다.

## SOAR Rule 구조 설명

SOAR(State, Operator, And Result)는 인지 아키텍처로, 에이전트가 **현재 상태(State)** 를 보고 **행동(Operator)** 을 제안하고 선택하여 **결과(Result)** 를 만들어내는 방식으로 동작합니다.

이 프로젝트에서는 카페 추천을 위해 다음과 같이 SOAR 개념을 적용합니다.

## Rule 구성 요소

### 1. Boilerplate (공통 구조)

모든 rule에 공통으로 들어가는 SOAR 상태 전파 구조입니다.

- `elaborate*top-state`: 최상위 state를 모든 하위 state에서 참조할 수 있도록 전파
- `elaborate*state*item*down`: 상위 state의 operator 후보를 하위 state로 복사 (tie-breaking 시 필요)
- `apply*recommend-cafe*send-to-python`: 최종 선택된 operator를 output-link로 Python에 전달

### 2. OPERATOR 제안

카페가 `operator_keywords` 중 하나라도 보유하면 추천 후보로 등록됩니다 (OR 조건).
여러 카페가 동시에 후보로 올라오면 tie(동점) 상태가 되어 tie-breaking 단계로 넘어갑니다.

```soar
sp {recommend*OPERATOR*넓은공간
   (<c> ^keyword << |실내공간_넓다| |카페_넓다| |매장_넓다| ... >>)
-->
   (<s> ^operator <o> +)
}
```

### 3. S1 ~ SN Judge (단계별 tie-breaking)

`tiebreak_keywords`의 순서대로 우열을 가립니다.

- S1: tiebreak_keywords[0] 보유 카페 우선
- S2: 여전히 tie면 tiebreak_keywords[1] 보유 카페 우선
- S3, S4, S5... 순서로 계속 진행

tie-breaking은 SOAR의 **impasse(교착 상태)** 메커니즘을 활용합니다. tie가 발생하면 자동으로 서브 state가 생성되고, 해당 서브 state에서 다음 우선순위 조건으로 판단합니다.

```soar
sp {recommend*S1*judge*넓은공간
   (<c1> ^keyword |3층_엄청 넓다|)
   - { (<c2> ^keyword |3층_엄청 넓다|) }
-->
   (<s> ^operator <o1> > <o2>)
}
```

### 4. LLM Fallback

모든 tie-breaking 단계를 거쳐도 동점이면 OpenAI API를 호출하여 최종 판단을 요청합니다.
`stopped-depth`는 몇 단계까지 rule-based 판단을 시도했는지를 나타내며, XAI 설명에 활용됩니다.
`tiebreak_keywords`가 없는 rule은 OPERATOR 제안 직후 바로 LLM fallback으로 넘어갑니다.

```soar
sp {resolve*tie*S6*ask-llm*넓은공간
   ...
-->
   (<ol> ^ask-llm <req>)
   (<req> ^cand1 <c1-name> ^cand2 <c2-name> ^stopped-depth 6)
}
```

## 실제 추론에서의 활용

`.soar` 파일의 논리 구조는 Python SOAR 추론 엔진에서 아래와 같이 동작합니다.

```
질문 입력
→ BGE-M3로 OpenSearch에서 유사 rule 검색
→ 해당 rule의 operator_keywords로 후보 카페 선정
→ tiebreak_keywords 순서대로 우열 판단
→ 최종 1개 카페 추천 + 판단 근거 반환 (XAI)
```

## 실행 방법

```bash
cd soar/Make_Rule
python generate_rule.py
```

## 결과

- 입력: `data/soar_rule_keywords.json` (252개)
- 출력: `soar/Make_Rule/soar_rules/` 디렉토리에 rule별 `.soar` 파일 생성
- 샘플: `soar/Make_Rule/soar_rules_sample/넓은공간.soar`