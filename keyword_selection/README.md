# Keyword Selection — sLLM Rule Selector

카페 추천 RAG의 **2단계**. 임베딩 검색이 뽑은 후보 rule 중에서 sLLM이 최종 1개를 선택(또는 `none`)합니다.

```
사용자 질문
   ↓
BGE-M3(파인튜닝) 임베딩 → OpenSearch KNN → 후보 rule top-15
   ↓
sLLM(Qwen2.5-1.5B QLoRA) → 정답 rule title 또는 "none"
   ↓
Soar 실행 → 매장 추천
```

---

## 핵심 설계 — 독립 조합 multi rule

이번 버전의 정확도를 끌어올린 핵심은 **rule 셋 재설계**입니다.

### 문제
기존 multi rule(`빈티지감성`=빈티지+감성, `쾌적향기`=향기+에어컨 등)이 single rule과 **동의어**였습니다. 검색 후보 15개에 `향기`·`힐링향기`·`쾌적향기` 같은 쌍둥이가 함께 떠서, 정답 라벨이 본질적으로 모호 → sLLM 정확도가 **0.727에서 천장**.

### 해결: multi = "독립적인 두 조건의 AND"
multi rule은 **둘 다 동시에 만족해야만 하는** 경우에만 가치가 있습니다.

| 조합 | 상관관계 | 한쪽만 추천하면 | multi 필요? |
|---|---|---|---|
| `여유` + `조용함` | 높음(동의어) | 만족 | ❌ single로 충분 |
| `주차장` + `공부` | 없음(독립) | 불만족 | ✅ AND 필요 |
| `반려동물` + `디저트` | 없음(독립) | 불만족 | ✅ |

→ 두 속성이 **멀수록(독립적일수록)** 좋은 multi. 가까운(동의어) 조합은 single 하나로 커버되니 제외.

### 생성 방식 (`build_multi_rules.py`)
1. 기존 multi 전부 폐기, **single 97개만 재료**로 사용
2. single description 임베딩 유사도로 **동의어 쌍 사전 차단** (보조)
3. **LLM이 "독립적 + 현실적(동시 요구 가능) + 모순 아님"** 판단 → 조합 생성
4. 결과: 독립 multi 80개 (멤버쌍 유사도 평균 0.61)

```
rule 180(동의어 multi 83) → 177(single 97 + 독립 multi 80)
```

---

## 데이터 파이프라인

```
data/rule_metadata_merged.json (177 rule)
   ↓ generate_keyword_questions_data.py   rule당 20개 질문 + 메뉴-복합 증강 20%
keyword_questions.jsonl (4248개) / test.jsonl (389개, gpt-4o)
   ↓ generate_finetune_data.py            질문 → 임베딩 → OpenSearch top-15 후보
finetune_keyword_selection.jsonl          정답이 top-15에 있으면 title, 없으면 "none"
   ↓ split_data.py                        title-stratified 9:1
train.jsonl (3722) / val.jsonl (526) / test_with_candidates.jsonl (389)
```

- **메뉴-복합 증강**: 실제 질문은 메뉴+비메뉴가 섞임("라떼 맛집인데 조용한 카페"). 메뉴 텍스트를 노이즈로 무시하고 비메뉴 rule을 고르도록 20% 주입 (gold=비메뉴 rule).
- **위치 편향 방지**: 후보 15개를 랜덤 셔플 후 rank 재부여 → 정답이 항상 다른 위치에.
- **none**: 정답이 top-15 밖이면 `none` 라벨 (train 5.2%). 별도 생성 없이 검색 miss로 자연 발생.

| 단계 | match(정답 후보 내) | none |
|---|---|---|
| finetune train | 94.9% | 5.1% |
| finetune test | 99.2% | 0.8% |

---

## 모델 / 학습

- **모델**: Qwen2.5-1.5B-Instruct (EXAONE은 Unsloth 미지원으로 제외)
- **프레임워크**: Unsloth (TRL 기반, 커널 최적화로 속도 2배·메모리 60%↓)
- **방식**: QLoRA 4bit (선택형 단순 task라 양자화 손실 제한적)
- **설정**: epoch 3, batch 4 × grad_accum 4, lr 2e-4, max_seq 2048
- **GPU**: RTX 4090 1x (RunPod)

```bash
cd keyword_selection
uv sync
uv run python train.py    --config config/qwen_1.5b.yaml
uv run python evaluate.py  --config config/qwen_1.5b.yaml
uv run python upload.py    --config config/qwen_1.5b.yaml
```

---

## 평가 결과

| 지표 | 동의어 multi (이전) | **독립 multi (현재)** | Δ |
|---|:-:|:-:|:-:|
| 전체 Accuracy | 0.727 | **0.884** | +15.7%p |
| match Accuracy | 0.732 | **0.889** | +15.7%p |

- rule 중첩 해소 → finetune match율 87%→95% (후보에 정답이 더 자주 포함) + 후보에 쌍둥이 rule이 안 떠 sLLM 혼동 감소 → 두 효과가 곱해져 +15.7%p.
- none(test 3개)은 표본이 작아 무의미.
- sLLM이 못 거른 케이스는 다음 단계 Soar fallback이 보완.

---

## OpenSearch

```bash
# 로컬 Docker
docker run -d --name opensearch -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=your_password" \
  opensearchproject/opensearch:2.13.0

# rule 인덱싱 (177 rule, description → 1024d 벡터 → cafe_rules 인덱스)
uv run python indexing.py
```

> sLLM 학습 자체는 OpenSearch가 필요 없습니다(후보가 jsonl에 저장됨). 인덱싱은 데이터 생성 시에만 사용.

---

## 파일 구조

```
keyword_selection/
├── config/qwen_{1.5b,3b,7b}.yaml      # 학습 설정
├── data/
│   ├── train.jsonl / val.jsonl        # 학습/검증 (후보 15개 + 정답)
│   ├── test_with_candidates.jsonl     # 평가
│   ├── keyword_questions.jsonl        # 생성 질문(train)
│   ├── test.jsonl                     # 생성 질문(test)
│   └── archive/                       # v1 산출물
├── indexing.py                        # OpenSearch 인덱싱 (BGEM3FlagModel)
├── generate_keyword_questions_data.py # 학습 질문 + 메뉴 증강 생성
├── generate_test_data.py              # test 질문 생성 (gpt-4o)
├── generate_finetune_data.py          # 질문→top-15 후보→정답/none
├── split_data.py                      # train/val 분할
├── prompt.py                          # sLLM instruction 템플릿
├── train.py / evaluate.py / upload.py # 학습 / 평가 / HF 업로드
└── pyproject.toml                     # Unsloth + TRL 환경

# multi rule 생성은 data_collection/data_keyword_select/build_multi_rules.py
```

---

## HuggingFace

| 항목 | 링크 |
|---|---|
| 키워드 선택 모델 | `east-right/cafe-keyword-selection-qwen-1.5b` (private) |
| 임베딩 모델 | `east-right/bge-m3-cafe-finetuned` (private) |
