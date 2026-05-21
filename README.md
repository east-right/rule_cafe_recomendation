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
<img width="778" height="892" alt="soar 아키텍처 현황" src="https://github.com/user-attachments/assets/1d146415-7ddb-42d8-8e26-d3a6896af052" />

1. 질문이 들어오면 `select_keyword`에서 질문과 관련이 높은 키워드 선택
2. 선택된 키워드에 해당하는 soar 추천 rule 선택

   2.1 만약 키워드에 해당하는 rule이 존재하지 않으면 LLM을 활용하여 새로운 rule 생성
4. rule을 사용한 추천 결과가 하나가 아니라면(교착 상태)면 LLM을 활용하여 추가적인 rule 생성
5. 추천 결과가 반환이 되었을 때 최종 result에 카페명 저장

   4.1. 만약 result에 저장된 결과가 2개가 아니라면 추천된 매장 제외 재추천 실행
7. 추천 매장이 두 개면 종료
---

## 3. 고도화 목표(기술 스택)
### 3.1 시스템 재정비 및 오케스트레이션 기능 추가
1. 최종 답변 모델 생성및 추가(openapi)

   1.1. 매장의 정보를 요약한 데이터 베이스 구축(mysql, postgreSQL)

   1.2. opensearch를 사용한 최종 답변 rag 시스템 구축(opensearch, bm25, bgem3)
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
<img width="1452" height="301" alt="image" src="https://github.com/user-attachments/assets/e3b08238-678b-4df7-af2b-a1052f229cb6" />

