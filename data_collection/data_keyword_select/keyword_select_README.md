# Data Keyword Select Branch — `feat/keyword-select`

카페 추천 시스템의 SOAR Rule 선택을 위한 Rule 메타데이터 생성 파이프라인입니다.
사용자 질문이 들어왔을 때 OpenSearch RAG와 키워드 선택 모델로 적절한 Rule을 선택하기 위한 데이터를 구축합니다.

---

## 전체 파이프라인 개요

```
valid_questions.json (유효 질문 3000개)
    ↓
[STEP 1] Single Rule 생성 (generate_rule.py)
    ↓
[STEP 2] Title 그룹핑 및 병합 (merge_rules.py)
    ↓
[STEP 3] 수동 패치 (patch_rules.py)
    ↓
[STEP 4] Multi Rule 생성 (generate_rule_multi.py)
    ↓
[STEP 5] 파인튜닝 데이터 생성
    ↓
rule_metadata_merged.json (313개 rule) ★
finetune_embd_query_pos.json (1136개) ★
```

---

## STEP 1. Single Rule 생성

**파일:** `data_collection/data_keyword_select/generate_rule.py`
**산출물:** `data/rule_metadata.json` (801개)

`valid_questions.json`의 `type=single` 질문 1654개를 대상으로 GPT Batch API를 사용해 각 질문에 대한 Rule 메타데이터를 생성합니다.

| 필드 | 설명 |
|------|------|
| `title` | Rule 제목 (2~6자) |
| `description` | RAG 벡터 검색용 Rule 설명 (1~2문장) |
| `keywords` | `market_item.csv`의 `최종_키워드`에서 선택한 키워드 목록 |

**키워드 컨텍스트:** `market_item.csv`의 ATMOSPHERE/FACILITY/TARGET 긍정 키워드 900개를 전체 제공합니다. LLM이 질문과 관련된 키워드를 선택합니다.

**중복 병합:** Jaccard 유사도 0.4 기준으로 keywords가 유사한 Rule을 병합합니다.

---

## STEP 2. Title 그룹핑 및 병합

**파일:** `data_collection/data_keyword_select/merge_rules.py`
**산출물:** `data/rule_metadata_merged.json` (125개), `data/title_groups.json`

801개 Rule의 335개 unique title을 `gpt-4.1`로 의미 기반 그룹핑합니다.

```
편안함 / 편안한 / 편안 / 편안한 분위기 → "편안함"
공부 / 카공 / 작업 / 집중 / 스터디 → "집중"
```

각 그룹에 `confidence`(확신/애매)와 `reason`을 부여하여 검토를 용이하게 합니다.

---

## STEP 3. 수동 패치

**파일:** `data_collection/data_keyword_select/patch_rules.py`

불필요하거나 모호한 Rule을 삭제/통합합니다.

```
삭제: 여름, 간식, 위치좋은, 대기, 커피 등
통합: 특별한분위기 → 카페분위기
rename: 산책 → 숲속카페
```

최종 단일 Rule: **118개**

---

## STEP 4. Multi Rule 생성

**파일:** `data_collection/data_keyword_select/generate_rule_multi.py`
**산출물:** `data/rule_metadata_merged.json`에 추가

`valid_questions.json`의 `type=multi` 질문 중 single rule로 조합 가능한 389개를 대상으로 multi Rule을 생성합니다.

**조합 방식:**
```
multi 질문의 seed_keywords
    ↓
각 keyword가 속한 single rule의 keywords 합집합 → 컨텍스트 제공
    ↓
LLM이 title + description + keywords 선택
```

title 3글자 이하 또는 single rule title과 중복되는 194개 삭제 후 **195개** 유지.

**최종 Rule 수: 313개 (single 118개 + multi 195개)**

---

## STEP 5. 파인튜닝 데이터 생성

각 Rule의 `question` + `merged_questions`를 query로, `description`을 positive로 사용합니다.

```
313개 rule
    ↓
question (313개) + merged_questions (823개)
    ↓
총 1136개 query-pos 쌍
    ↓
finetune_embd_query_pos.json ★
```

---

## 최종 산출물

| 파일 | 설명 |
|------|------|
| `data/rule_metadata_merged.json` | 313개 Rule (title/description/keywords) |
| `data/finetune_embd_query_pos.json` | 1136개 query-pos 쌍 (임베딩 파인튜닝용) |

---

## Rule 구조

```json
{
    "title":            "집중",
    "description":      "조용하고 집중하기 좋은 환경을 원하는 카공족을 위한 카페를 추천합니다.",
    "keywords":         ["카공", "콘센트_잘 구비되어있다", "조용한 카페"],
    "question":         "카공하기 좋은 카페 추천해 줄 수 있어요?",
    "merged_questions": ["공부하기 좋은 카페 어디 있을까요?", ...],
    "confidence":       "확신",
    "reason":           "카공·공부·집중에 적합한 카페라는 공통점"
}
```

---

## 메뉴 처리 방식

MENU 키워드는 Rule로 만들지 않고 runtime에 처리합니다.

```
메뉴 단독 질문: "아메리카노 맛있는 카페"
→ 질문에서 메뉴명 파싱
→ market_item.csv MENU 긍정 키워드에서 tfidf_score 기반 numeric preference
→ SOAR rule 즉석 생성

메뉴 + 비메뉴 복합: "아메리카노 맛있고 조용한 카페"
→ 메뉴: 후보 매장 필터링 (WM에 올리기 전)
→ 비메뉴: RAG로 rule 선택
```

---

## 파일 구조

```
data_collection/data_keyword_select/
├── generate_rule.py        # Single Rule 생성
├── generate_rule_multi.py  # Multi Rule 생성
├── merge_rules.py          # Title 그룹핑 및 병합
├── patch_rules.py          # 수동 패치
└── prompt.py               # 프롬프트 관리
```
