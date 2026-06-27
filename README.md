# ☕ XAI 카페 추천 시스템

> 인간의 인지 아키텍처(Soar)와 대형 언어 모델(LLM)을 결합한 설명 가능한(XAI) 카페 추천 에이전트

---

## 1. 프로젝트 목적 및 목표 (Project Objectives)

기존의 딥러닝 기반 추천 시스템은 높은 정확도를 보이지만 추천의 '이유'를 설명하지 못하는 **블랙박스(Black-box) 한계**가 있다. 본 프로젝트는 이 문제를 해결하고, 사용자의 복합적이고 암묵적인 요구사항까지 정밀하게 타겟팅하는 것을 목표로 한다.

- **설명가능한(XAI) 추천 시스템 구현**: 인지 아키텍쳐인 soar를 활용하여 사용자의 질문(요구사항)에 기반한 카페 추천하고 추천에 대한 근거를 제공
- **사용자의 지역에 기반한 추천**: 현재는 "관악구 신림동"에 대한 카페만 추천 진행
> 해당 시스템의 Main Reference: [CogRec: A Cognitive Recommender Agent Fusing Large Language Models and Soar for Explainable Recommendation](https://arxiv.org/abs/2512.24113)
---

## 2. 현 상황 아키텍처
<img width="1424" height="564" alt="image" src="https://github.com/user-attachments/assets/f18c3301-787f-4d8f-87a7-68f7fc12daa8" />

1. 질문이 들어오면 유효성 검증
2. 키워드 선택 sLLM에게 Rule 선택 요청
  - 모든 룰을 제시하기 보단 Rag를 활용하여 관련성 높은 Rule만 제시
  - 만약 선택이 가능한 Rule이 없다면 LLM을 활용하여 신규 룰 생성
3. Soar를 활용하여 Rule-base 추천을 진행
   - 교착상태(Impass)에 들어가면 LLM을 활용하여 추가 룰 업데이트
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

# 2) OpenSearch 띄우고 rule 인덱싱
docker compose up -d
uv run python keyword_selection/indexing.py    # cafe_rules 인덱스 생성

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

