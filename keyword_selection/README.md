# Keyword Selection Model

카페 추천 시스템의 키워드 선택 모델 파인튜닝 모듈입니다.  
사용자 질문이 들어오면 OpenSearch RAG로 후보 rule 10개를 검색하고, sLLM이 가장 적합한 rule을 선택합니다.

---

## 아키텍처

```
사용자 질문
    ↓
BGE-M3 (파인튜닝) 임베딩
    ↓
OpenSearch KNN 검색 → 후보 rule 10개
    ↓
sLLM (파인튜닝) → 정답 rule title 또는 "none"
```

---

## 파일 구조

```
keyword_selection/
├── config/
│   ├── qwen_1.5b.yaml              # Qwen2.5-1.5B 학습 설정
│   ├── qwen_3b.yaml                # Qwen2.5-3B 학습 설정
│   └── qwen_7b.yaml                # Qwen2.5-7B 학습 설정
├── data/
│   ├── finetune_keyword_selection.jsonl  # 전체 파인튜닝 데이터 (1136개)
│   ├── train.jsonl                 # 학습 데이터 (1021개)
│   └── test.jsonl                  # 평가 데이터 (115개)
├── compare_search.py               # Brute-force vs KNN 검색 방식 비교
├── evaluate.py                     # 파인튜닝 모델 평가
├── generate_finetune_data.py       # 파인튜닝 데이터 생성
├── indexing.py                     # OpenSearch rule 인덱싱
├── prompt.py                       # sLLM instruction 템플릿
├── search.py                       # OpenSearch KNN 검색
├── setup.sh                        # RunPod 환경 설정 및 학습 실행
├── split_data.py                   # train/test 분할
├── train.py                        # Unsloth SFT 학습 스크립트
├── upload.py                       # HuggingFace 업로드
└── pyproject.toml                  # 파인튜닝 전용 의존성
```

---

## OpenSearch 구축

### 로컬 Docker 실행
```bash
docker run -d --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=your_password" \
  opensearchproject/opensearch:2.13.0
```

### rule 인덱싱
```bash
python keyword_selection/indexing.py
```
- `data/rule_metadata_merged.json`에서 313개 rule 로드
- `east-right/bge-m3-cafe-finetuned` 모델로 description 임베딩 (1024차원)
- OpenSearch `cafe_rules` 인덱스에 업로드

### 검색 방식 검증
```bash
python keyword_selection/compare_search.py
```

| K | Brute-force | KNN | Δ |
|---|---|---|---|
| Recall@1 | 0.5439 | 0.5439 | 0.0 |
| Recall@5 | 0.8246 | 0.8246 | 0.0 |
| Recall@10 | 0.8991 | 0.8991 | 0.0 |

→ 313개 스케일에서 KNN과 Brute-force 결과 완전 동일 확인. **KNN 채택.**

---

## 파인튜닝 데이터

### 생성
```bash
python keyword_selection/generate_finetune_data.py
python keyword_selection/split_data.py
```

### 데이터 구조
```json
{
    "query": "조용하고 콘센트 있는 카페 추천해줘",
    "candidates": [
        {"rank": 3, "title": "카공족", "description": "..."},
        {"rank": 1, "title": "넓은공간", "description": "..."},
        ...
    ],
    "answer": "카공족"
}
```

### 데이터 통계
| 구분 | 개수 |
|---|---|
| 전체 | 1136개 |
| 정답 있음 | 1053개 (92.7%) |
| 정답 없음 (none) | 83개 (7.3%) |
| train | 1021개 |
| test | 115개 |

**위치 편향 방지:** candidates 순서를 랜덤 셔플하여 정답이 항상 다른 rank에 위치하도록 처리

---

## 모델 선정

### 후보 모델
EXAONE-3.5 (LG AI Research)와 Qwen2.5 (Alibaba) 두 계열을 검토했으나, **EXAONE은 Unsloth 미지원**으로 인해 Qwen2.5로 결정. EXAONE은 추후 TRL+PEFT 방식으로 별도 실험 예정.

Qwen2.5 계열에서 1.5B / 3B / 7B 세 가지 사이즈를 비교하여 최적 모델 선정.

### 파인튜닝 패키지: Unsloth
TRL+PEFT, LLaMA-Factory, Unsloth 세 가지를 검토한 결과 **Unsloth 채택**.

- **TRL+PEFT**: 가장 정통적인 방식이나 속도/메모리 최적화 없음
- **LLaMA-Factory**: 편의성은 높으나 커스텀 데이터 형식 적용이 번거로움
- **Unsloth**: TRL 기반이면서 커널 최적화로 학습 속도 2배, 메모리 60% 절약. RunPod 비용 최적화에 유리

### 학습 방식: QLoRA
배치 크기 축소로 LoRA도 가능했으나, 동일 VRAM(RTX 4090 24GB)에서 더 큰 배치로 안정적인 학습이 가능한 QLoRA(4bit) 선택. 선택형 단순 task 특성상 4bit 양자화로 인한 성능 손실이 제한적일 것으로 판단.

---

## 파인튜닝

### 환경
- **프레임워크:** Unsloth (TRL 기반)
- **방식:** QLoRA (4bit 양자화, 기울기 누적 적용)
- **GPU:** RTX 4090 1x (RunPod)

### RunPod 실행
```bash
git clone -b feat/keyword-selection https://github.com/east-right/cafe_recomendation.git /workspace/cafe_recomendation
cd /workspace/cafe_recomendation
echo "HUGGINGFACE_TOKEN_WRITE=your_token" > .env
cd keyword_selection
uv sync
uv run python train.py --config config/qwen_1.5b.yaml
```

---

## 평가 결과

```bash
python keyword_selection/evaluate.py --config config/qwen_1.5b.yaml
```

| 모델 | 전체 Accuracy | match Accuracy | none Accuracy | 평균 추론 시간 |
|---|---|---|---|---|
| Qwen2.5-1.5B | **0.4870** | **0.5189** | **0.1111** | **0.2937s** |
| Qwen2.5-3B | 0.4609 | 0.5000 | 0.0000 | 0.3752s |
| Qwen2.5-7B | 0.4609 | 0.5000 | 0.0000 | 0.3150s |

→ 모델 크기가 커져도 성능 향상 없음. **Qwen2.5-1.5B 채택.**

### 성능이 낮은 이유
현재 학습 데이터(1021개)의 품질 문제로 판단. 구체적으로:
- `finetune_embd_query_pos.json` 기반으로 생성한 데이터라 질문-rule 매핑의 다양성이 부족
- none 케이스 83개(7.3%)로 fallback 학습이 부족
- 향후 데이터 보강 및 재학습을 통해 성능 개선 예정

---

## HuggingFace

| 항목 | 링크 |
|---|---|
| 키워드 선택 모델 (1.5B) | east-right/cafe-keyword-selection-qwen-1.5b |
| 임베딩 모델 | east-right/bge-m3-cafe-finetuned |

---

## .env 설정

```dotenv
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=your_password
HUGGINGFACE_TOKEN_READ=hf_xxx
HUGGINGFACE_TOKEN_WRITE=hf_xxx
```