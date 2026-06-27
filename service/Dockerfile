# 앱 서버 (LangGraph 오케스트레이션) 이미지 — 가볍다 (모델/torch 없음).
# 추론은 모델서버(:8001)에 HTTP로 위임, 검색은 OpenSearch(:9200).
# Soar(Windows 바이너리)를 제거해서 이제 리눅스 컨테이너화가 가능해졌다.
#
# 빌드 컨텍스트 = repo 루트:  docker build -f service/Dockerfile -t cafe-app .

FROM python:3.12-slim

WORKDIR /app

# 1) 의존성 (캐시 레이어)
COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

# 2) 앱 코드 + 의존 모듈(OpenSearch 클라/질문 분류기) + 시드 DB
COPY service/ ./service/
COPY keyword_selection/__init__.py keyword_selection/search.py ./keyword_selection/
COPY question_classifier/__init__.py question_classifier/classify.py ./question_classifier/
COPY data/cafe.db ./data/cafe.db

EXPOSE 8000
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
