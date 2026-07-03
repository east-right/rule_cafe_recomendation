# 모델 추론 서버 (CPU 백엔드) 이미지.
# 빌드 컨텍스트는 repo 루트:  docker build -f inference/Dockerfile -t cafe-inference .
#
# 모델(BGE-M3, Qwen gguf)은 이미지에 굽지 않고 런타임에 HF에서 받는다.
#   docker run -e HUGGINGFACE_TOKEN_READ=hf_xxx -p 8001:8001 cafe-inference

FROM python:3.12-slim

# llama-cpp-python 소스 빌드용 (prebuilt wheel이 일부 CPU에서 illegal instruction)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) 일반 의존성 (캐시 레이어)
COPY inference/requirements.txt ./inference/requirements.txt
RUN pip install --no-cache-dir -r inference/requirements.txt

# 2) llama-cpp-python: AVX2로 빌드, AVX512 비활성 (이식성)
#    AVX512가 보장된 호스트/클라우드라면 prebuilt wheel로 바꿔 빌드 시간을 줄여도 된다.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF"
RUN pip install --no-cache-dir llama-cpp-python

# 3) 앱 코드 (자주 바뀌므로 마지막)
COPY inference/ ./inference/

EXPOSE 8001
CMD ["uvicorn", "inference.server:app", "--host", "0.0.0.0", "--port", "8001"]
