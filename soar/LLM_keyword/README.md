# SOAR Rule Keyword Extraction

## 개요
SOAR 아키텍처 기반 카페 추천 시스템에서 rule을 실행하려면 각 rule이 **어떤 카페를 후보로 올릴지(operator)**, **후보 간 우열을 어떻게 가릴지(tiebreak)** 를 정의해야 합니다.

이 모듈은 사전에 구축된 rule 메타데이터(`rule_metadata_merged.json`)에서 rule title과 keyword 풀을 읽어, OpenAI API를 통해 각 rule에서 가장 핵심이 되는 keyword를 자동으로 선별합니다.

선별된 keyword는 이후 `soar/Make_Rule` 모듈에서 실제 `.soar` 파일로 변환되어 SOAR 추론 엔진에 적재됩니다.

## keyword 선별 기준
- `operator_keywords` (4~5개): rule의 본질을 가장 직접적으로 표현하는 keyword. 이 중 하나라도 보유한 카페가 추천 후보로 올라갑니다 (OR 조건).
- `tiebreak_keywords` (4~5개): 후보 카페 간 우열을 가리는 keyword. 리스트 순서가 우선순위이며 index 0이 S1(1차 tie-breaking), index 1이 S2(2차 tie-breaking)로 사용됩니다.

## 출력 구조
```json
  {
    "title": "편안함",
    "operator_keywords": ["편안한 카페","편안한 분위기","좌석_넓고 편하다","아늑한 카페","휴식"],
    "tiebreak_keywords": ["좌석_편하다","좌석 간격_넓다","카페 인테리어_편안하다","넓고 쾌적한 공간","소파자리_좋다"]
  },
```

## 실행 방법
```bash
python keyword_extraction.py
```

## 결과
- 입력: 253개 rule (중복 title 합집합 처리 후)
- 성공: 250개
- 출력: `data/soar_rule_keywords.json`

## 환경변수
루트 `.env`에 `OPENAI_API_KEY` 필요