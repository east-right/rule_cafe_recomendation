# Embedding — `feat/rag-system-v2`

카페 추천 RAG의 **Rule 검색용 BGE-M3 임베딩 파인튜닝** 파이프라인입니다.
v2 정규화 데이터로 rule 셋을 재생성하고, 도메인 특화 + 하드 네거티브로 검색 성능을 끌어올렸습니다.

```
질문 입력
   ↓
[임베딩] BGE-M3 파인튜닝 ──→ OpenSearch KNN ──→ Rule 후보 top-k
   ↓
[sLLM] 후보 중 최적 Rule 선택 (도메인 파인튜닝 = 사실상 리랭커)
   ↓
Soar 실행 → 매장 추천
```

---

## 파인튜닝 결과 (test 501개 / rule 180개)

| 지표 | dragonkue/BGE-m3-ko | 파인튜닝 후 | 향상 |
|---|:-:|:-:|:-:|
| Recall@1 | 0.281 | 0.545 | +0.264 |
| Recall@5 | 0.519 | 0.822 | +0.303 |
| **Recall@10** | 0.619 | **0.888** | +0.269 |
| Recall@15 | 0.647 | 0.930 | +0.283 |
| Recall@20 | 0.679 | **0.962** | +0.284 |

- **정답 rule의 96%가 top-20 안**에 있음 → 후보를 넓게(top-15~20) 뽑아 sLLM에 넘기면 recall 회수 가능
- 누수 없는 정직한 수치 (메뉴-복합 증강을 split 이후 train/test 각각 적용)

---

## 데이터 파이프라인

```
data/rule_metadata_merged.json (180 rule = single 97 + multi 83)   ← v2 정규화 market_item에서 재생성
   ↓  build_embd_data.py
각 rule: question + merged_questions → query / description → pos
   = query-pos 2002쌍 (finetune_embd_query_pos.json)
   ↓  finetune_bge.py
[1] confidence stratify로 80/20 split (seed 42)
[2] 메뉴-복합 증강: split 후 train·test 각각 25% (gold=비메뉴 rule 그대로) ← 누수 방지
[3] 하드 네거티브 채굴: 베이스 모델로 query별 가장 헷갈리는 rule 7개 → neg
   ↓
InfoNCE 학습 (train_group_size = 1 pos + 7 neg)
```

### 핵심 설계
- **하드 네거티브**: easy random neg 대신 헷갈리는 rule을 neg로 → 유의어 rule 미세 분리 학습 (v1에선 데이터 품질 문제로 보류했던 것)
- **메뉴-복합 증강**: 실제 파이프라인은 질문을 통째로 RAG에 넣으므로("아이스아메리카노 맛집인데 조용한 카페"), 메뉴 텍스트를 노이즈로 무시하고 비메뉴 rule을 찾도록 학습. **split 이후** 적용해 train↔test 누수 차단.

---

## 파인튜닝 설정

| 항목 | 값 |
|---|---|
| 베이스 모델 | `BAAI/bge-m3` |
| Epoch | 5 |
| Batch | 16 |
| LR | 1e-5 |
| Temperature | 0.02 |
| Max Length | 512 |
| Loss | InfoNCE |
| Hard Negative | 7개/query (train_group_size 8) |
| GPU | NVIDIA A40 (cu128) |

---

## 실행 방법 (uv)

```bash
cd embedding
uv sync

# 파인튜닝 (split → 메뉴증강 → 하드네거 채굴 → 학습)
uv run torchrun --nproc_per_node=1 finetune_bge.py

# 평가 (Recall@1/5/10/15/20)
uv run python evaluate.py

# HF 업로드 (모델 + 학습 데이터)
uv run python upload.py
```

---

## 검사 / 실험 스크립트

| 스크립트 | 용도 | 결론 |
|---|---|---|
| `inspect_retrieval.py` | 질문→top-10 확인 + test 오답 케이스 출력 | 오답 상당수는 정답 라벨 오류(모델이 오히려 맞음) |
| `evaluate_hybrid.py` | dense + sparse(lexical) 결합 측정 | **이득 없음**(+0.5pt) → description이 generic이라 lexical 무력 → dense 단독 확정 |
| `evaluate_rerank.py` | top-20 → bge-reranker-v2-m3 재정렬 | **base 리랭커는 더 나쁨**(0.89→0.77) → 도메인 미학습. 도메인 리랭커는 곧 sLLM이라 별도 리랭커 불필요 |

```bash
uv run python inspect_retrieval.py                          # 오답 케이스
uv run python inspect_retrieval.py "마카롱 맛집인데 조용한 카페"  # 직접 질문
uv run python evaluate_hybrid.py
uv run python evaluate_rerank.py
```

---

## 파일 구조

```
embedding/
├── finetune_bge.py        # split + 메뉴증강 + 하드네거 채굴 + 학습
├── evaluate.py            # Recall@1/5/10/15/20
├── evaluate_hybrid.py     # 하이브리드 실험 (결론: dense 단독)
├── evaluate_rerank.py     # 리랭커 실험 (결론: base 리랭커 X)
├── inspect_retrieval.py   # 검색 검사
├── upload.py              # HF 업로드
├── pyproject.toml         # uv 환경 (torch cu128, FlagEmbedding 등)
├── data/
│   ├── finetune_train/train.jsonl   # 학습 데이터 (런타임 생성)
│   └── test_query_pos.json          # 평가 데이터 (런타임 생성)
└── models/bge-m3-cafe/    # 파인튜닝 모델 (gitignore)
```

---

## 남은 과제 (후속 sLLM/인덱싱 브랜치)
- **후보 확대로 recall 회수**: top-20 안에 정답 96% → sLLM을 top-15~20 후보로 학습하면 0.888 → 0.93~0.96 회수 (별도 리랭커 불필요)
- **문체 쏠림**: 학습 질문이 문어체 70% / 구어체 20% → 실제 구어체 대응 과소평가. 구어체 질문 보강 검토
- **description distinctive화**: generic("~분들을 위한 추천")이라 유의어 rule 1위 경쟁(R@1 0.55) → 고유 특징 명시하면 dense·sparse 둘 다 개선
- **메뉴-복합 처리**: 현재는 임베딩이 메뉴 노이즈를 무시하도록 학습. sLLM 단계에서도 gold=비메뉴 정합성 유지 필요
