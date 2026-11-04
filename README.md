# ☕ XAI 카페 추천 시스템

> Soar 인지 아키텍처에서 영감받은 규칙기반 추론과 대형 언어 모델(LLM)을 결합한 설명 가능한(XAI) 카페 추천 에이전트

---

## 1. 프로젝트 목적 및 목표 (Project Objectives)

기존의 딥러닝 기반 추천 시스템은 높은 정확도를 보이지만 추천의 '이유'를 설명하지 못하는 **블랙박스(Black-box) 한계**가 있다. 본 프로젝트는 이 문제를 해결하고, 사용자의 복합적이고 암묵적인 요구사항까지 정밀하게 타겟팅하는 것을 목표로 한다.

- **설명가능한(XAI) 추천 시스템 구현**: Soar 인지 아키텍처에서 영감받은 규칙기반 trace 추론으로 사용자의 질문(요구사항)에 맞는 카페를 추천하고, 추천에 이른 단계(후보 → 부정 필터 → tiebreak narrowing)를 근거로 제공
- **사용자의 지역에 기반한 추천**: 현재는 "관악구 신림동"에 대한 카페만 추천 진행
- **구현 현황**: 현재(트랙1)는 Soar에서 영감받은 Python 규칙기반 trace로 XAI 추론을 구현한 **CPU 배포본**이다. 정식 Soar 인지 아키텍처(프로덕션 룰 + 상주 커널 + chunking 학습) 적용은 트랙2 로드맵.
> 해당 시스템의 Main Reference: [CogRec: A Cognitive Recommender Agent Fusing Large Language Models and Soar for Explainable Recommendation](https://arxiv.org/abs/2512.24113)
---

## 2. 현 상황 아키텍처
<img width="1424" height="564" alt="image" src="https://github.com/user-attachments/assets/f18c3301-787f-4d8f-87a7-68f7fc12daa8" />

1. 질문이 들어오면 유효성 검증
2. 키워드 선택 sLLM에게 Rule 선택 요청
  - 모든 룰을 제시하기 보단 Rag를 활용하여 관련성 높은 Rule만 제시
  - 만약 선택이 가능한 Rule이 없다면 LLM을 활용하여 신규 룰 생성
3. 규칙기반 trace로 Rule-base 추천 추론 진행 (Soar 인지 아키텍처에서 영감받은 투명한 단계별 narrowing)
   - 동점(impasse)이 되면 LLM으로 추가 구별 키워드를 받아 해소하고, 그 결과를 룰에 반영(학습)
4. 최종 선택된 매장의 정보를 활용하여 최종 추천 답변 생성
---

## 3. 고도화 목표(기술 스택)
3.1 데이터 생성  

    3.1.1 카페 리뷰 데이터 수집  
    3.1.2 카페 Item 데이터 구축(UMAP, HBSCAN, TF-IDF)  
    3.1.3 임베딩 모델 파인튜닝 데이터 구축  
    3.1.4 키워드 선택 모델 파인튜닝 데이터 구축  
    3.1.5 최종 답변 모델 - 매장 정보 데이터 구축  
3.2 모델 파인튜닝 및 파이프라인 구축  

    3.2.1 오픈소스 모델 기반 임베딩 모델 파인튜닝(BGE-M3)  
    3.2.2 오픈소스 모델 기반 키워드 선택 sLLM 파인튜닝  
    3.2.3 질문 유효성 검증 모델 파인튜닝(RoBerta)  
    3.2.4 Soar 추천 시스템 구축  
    3.2.5 최종 답변 파이프 라인 구축  
3.3 백엔드 엔지니어  

    3.3.1 fastapi+python을 활용한 백엔드 구축      
    3.3.2 파인튜닝 모델 vLLM 기반 서빙 인프라 구축     
    3.3.3 Redis 세션별 채팅 캐싱  
    3.3.4 Mem0를 활용한 유저 메모리 캐싱  
    3.3.5 langfuze를 활용한 LLMops  
    3.3.6 MVP 데모 화면 구축  
    3.3.7 기능별 레포지토리 정의  
---
## 4. WBS																													
<img width="1450" height="463" alt="image" src="https://github.com/user-attachments/assets/729b771a-2159-4cff-ab3f-8d48c3092b5f" />

---

## 5. 실행 방법 (Quick Start — CPU)

> 클론 후 로컬에서 추천 파이프라인을 띄우는 최소 절차. 3개가 떠야 한다:
> **OpenSearch(:9200)** · **모델 추론 서버(:8001)** · **앱 서버(:8000)**

**사전 준비**: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker

```bash
# 1) 클론 + 의존성 + 환경
git clone https://github.com/f-lab-edu/cafe_recomendation.git
cd cafe_recomendation
uv sync
cp .env.example .env          # OPENAI_API_KEY, HUGGINGFACE_TOKEN_READ 등 입력

# 2) OpenSearch 띄우고 rule 인덱싱 (2단계 — 둘 다 필요!)
docker compose up -d
uv run python keyword_selection/indexing.py       # ① RAG 벡터(title/description)
uv run python keyword_selection/uodate_index.py   # ② operator/tiebreak 키워드 (이거 빠지면 추천이 안 나옴)

# 3) 모델 추론 서버 (BGE-M3 + sLLM gguf) — 별도 터미널
uv run uvicorn inference.server:app --port 8001

# 4) 앱 서버 — 또 다른 터미널
uv run uvicorn service.api:app --port 8000

# 5) 호출
curl -N -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"question":"카공하기 좋은 카페 추천해줘"}'
```

**참고**
- `data/cafe.db`는 **시드 데이터로 커밋**돼 있어 별도 적재 불필요. (재생성하려면 `uv run python service/db/loader.py`)
- sLLM gguf(`east-right/cafe-keyword-selection-qwen-1.5b`)는 모델서버가 HF에서 자동 다운로드. 일부 CPU(AVX512 비활성)에서 illegal instruction이면 AVX2 소스 빌드 필요 → `keyword_selection/README.md` 참고.
- 모델 서버는 인터페이스(`/embed`·`/select_rule`)가 고정돼 있어, GPU 배포 시 `MODEL_SERVER_URL`만 교체하면 된다.

---

## 6. 측정 결과 (Measurement Summary)

> 평가셋: 신림동 카페 **181곳** · 리뷰 기반 LLM-judge gold qrels(50문항) + Rule 선택 테스트(389문항). 모두 **오프라인 지표**(실사용 클릭 아님). 상세·재현 코드는 `eval_data/results/`.

### 6.1 Rule 선택 정확도 (389문항, 동일 프롬프트·모델만 교체)

| 모델 | Accuracy | 후보밖 출력 |
|---|---|---|
| gpt-4.1-mini (Structured Output, enum 강제) | **0.931** | 0 |
| **파인튜닝 sLLM** (Qwen2.5-1.5B QLoRA, gguf) | **0.877** | 2 |
| gpt-4.1-mini (일반) | 0.766 | 71 |
| Qwen2.5-1.5B 원본 (파인튜닝 X) | 0.427 | 60 |

→ 파인튜닝으로 **0.43 → 0.88**(약 2배). 범용 GPT는 출력 형식을 강제(Structured Output)해야 0.93이고, 일반 호출은 형식 미준수(후보밖 71건)로 0.77. 작은 로컬 sLLM이 **제약 없이도** 일반 GPT를 앞섬.

### 6.2 추천 품질 (50문항, vs 무작위 기준선)

| 지표 | 시스템 (Rule 선택 시) | 무작위 |
|---|---|---|
| Hit@1 | **0.89** | 0.40 |
| NDCG@3 | **0.65** | 0.33 |

### 6.3 설명 가능성 · 안정성 · 비용 · 지연

| 항목 | 값 |
|---|---|
| 답변 근거 충실도 (Trace 실제 만족 조건과 일치) | 0.38 → **1.00** (근거 분리 수정 후) |
| 같은 질문 반복 재현성 (Rule·매장) | **100%** (Rule 같으면 Trace 결정적) |
| 요청당 비용 | **~$0.003** (정상 경로는 로컬 sLLM+규칙, LLM은 분류·답변에만) |
| 지연 — sLLM Rule 선택 | **~3초** (CPU, 후보 10개) · GPU/vLLM 전환 시 단축 |
| 지연 — 규칙기반 추론(Soar trace) | **0.007초** (XAI 추론 자체는 사실상 무비용) |

> 한계: 오프라인 리뷰기반 judge 기준이며 실사용 반응은 미측정. 표현 변형 강건성(의미 같은 다른 표현)은 Rule 0.66 수준으로 개선 과제. 데이터 최신성(폐업 반영)은 설계 단계.

