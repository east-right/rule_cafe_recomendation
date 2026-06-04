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
│   ├── exaone_2.4b.yaml        # EXAONE-3.5-2.4B 학습 설정
│   ├── exaone_7.8b.yaml        # EXAONE-3.5-7.8B 학습 설정
│   └── qwen_7b.yaml            # Qwen2.5-7B 학습 설정
├── data/
│   ├── finetune_keyword_selection.jsonl  # 전체 파인튜닝 데이터 (1136개)
│   ├── train.jsonl             # 학습 데이터 (1021개)
│   └── test.jsonl              # 평가 데이터 (115개)
├── compare_search.py           # Brute-force vs KNN 검색 방식 비교
├── evaluate.py                 # 파인튜닝 모델 평가
├── generate_finetune_data.py   # 파인튜닝 데이터 생성
├── indexing.py                 # OpenSearch rule 인덱싱
├── prompt.py                   # sLLM instruction 템플릿
├── search.py                   # OpenSearch KNN 검색
├── setup.sh                    # RunPod 환경 설정 및 학습 실행
├── split_data.py               # train/test 분할
├── train.py                    # Unsloth SFT 학습 스크립트
├── upload.py                   # HuggingFace 업로드
└── pyproject.toml              # 파인튜닝 전용 의존성
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

## 모델 선정 계획

파인튜닝 후 성능 비교를 통해 최종 모델 선정 예정

```
1단계: EXAONE-3.5-2.4B vs EXAONE-3.5-7.8B → 사이즈 영향 확인
2단계: EXAONE-7.8B vs Qwen2.5-7B → 한국어 특화 vs 범용 비교
3단계: 최종 모델 선정
```

---

## 파인튜닝

### 환경
- **프레임워크:** Unsloth (TRL 기반, 2배 빠른 학습 + 60% 메모리 절약)
- **방식:** LoRA (QLoRA 불필요 - RTX 4090 24GB VRAM으로 충분)
- **GPU:** RTX 4090 1x (RunPod)

### RunPod 실행
```bash
git clone -b feat/keyword-selection https://github.com/east-right/cafe_recomendation.git /workspace/cafe_recomendation
cd /workspace/cafe_recomendation
echo "HUGGINGFACE_TOKEN_WRITE=your_token" > .env
bash keyword_selection/setup.sh
```

세 모델 순서대로 자동 학습 (예상 소요시간: 2.5~4시간)

---

## 평가

```bash
python keyword_selection/evaluate.py --config config/exaone_2.4b.yaml
```

### 평가 지표
- **전체 Accuracy**: 전체 정확도
- **match Accuracy**: 정답 있는 케이스 정확도
- **none Accuracy**: 정답 없는 케이스 정확도

> 학습 완료 후 결과 업데이트 예정

---

## HuggingFace 업로드

```bash
python keyword_selection/upload.py --config config/exaone_2.4b.yaml
```

| 모델 | HuggingFace repo |
|---|---|
| EXAONE 2.4B | east-right/cafe-keyword-selection-exaone-2.4b |
| EXAONE 7.8B | east-right/cafe-keyword-selection-exaone-7.8b |
| Qwen 7B | east-right/cafe-keyword-selection-qwen-7b |

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
