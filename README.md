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
### 3.1 시스템 재정비 및 오케스트레이션 기능 추가
1. 최종 답변 모델 생성및 추가(openapi)

   1.1. 매장의 정보를 요약한 데이터 베이스 구축(mysql, postgreSQL)

   1.2. opensearch를 사용한 rag 시스템 구축(opensearch, bm25, bgem3)
3. 멀티턴 시스템 구축

   2.1. Redis를 활용하여 각 세션별 대화 내용 캐싱(Redis)

   2.2. 만약 대화 내용을 장기 메모리로도 관리한다면 Mem0(맴제로) 혹은 S3 고려

### 3.2 모델 파이프라인 최적화 및 서빙
3. 오픈소스 모델 기반 키워드 선택 sLLM 파인튜닝(PEFT, Huggingface)

   3.1 파인튜닝용 데이터 증강(openapi)
5. 모델 파인튜닝 실험 및 로깅(Mlflow)
6. 파인튜닝된 모델 vLLM 기반 BentoML 서빙 인프라 구축(BentoML)

### 3.3 백엔드 엔지니어
6. fastapi+python을 활용한 백엔드 구축(fastapi)
7. 기능별 레포지토리 정의
8. langfuze를 활용한 LLMops(langfuze)
9. MVP 데모 화면 구축(strimlit, React(선택 확률 낮음))
---
## 4. WBS																													
<img width="1450" height="463" alt="image" src="https://github.com/user-attachments/assets/729b771a-2159-4cff-ab3f-8d48c3092b5f" />


