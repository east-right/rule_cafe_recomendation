# inference/ — 모델 추론 서버

임베딩(BGE-M3)과 sLLM(Qwen2.5-1.5B GGUF)만 로드해 HTTP로 노출하는 **모델 전용 서버**.
`service/`(LangGraph 앱)가 이 서버를 호출하고, OpenSearch 검색은 앱이 담당한다.

## 왜 분리했나

모델(GPU 교체 대상)과 인프라(OpenSearch)를 분리하는 게 목적.
- 앱(`service`)은 그대로 두고, 모델 백엔드만 갈아끼울 수 있음
- **GPU 트랙**에서는 이 서버의 sLLM 백엔드를 vLLM으로 교체(인터페이스 동일)

## 실행

```bash
uv run uvicorn inference.server:app --host 0.0.0.0 --port 8001
```

startup에 모델 2개를 상주 로드:
- 임베딩: `east-right/bge-m3-cafe-finetuned` (SentenceTransformer)
- sLLM: `qwen1.5b-cafe-q8_0.gguf` (llama.cpp, CPU) — 로컬 빌드 우선, 없으면 HF 다운로드

## 엔드포인트

| 메서드 | 경로 | 역할 |
|---|---|---|
| GET | `/health` | 상태·백엔드·로드된 모델 |
| POST | `/embed` | `{texts:[...]}` → 정규화 임베딩 벡터 (질문·룰 description 검색용) |
| POST | `/select_rule` | `{question, candidates}` → 선택된 룰 title 또는 `"none"` |

## 파일

| 파일 | 역할 |
|---|---|
| `server.py` | FastAPI 서버 (모델 로드 + 3개 엔드포인트) |
| `prompt.py` | 룰 선택 프롬프트 (`service`에 의존하지 않는 독립 프롬프트) |

## 의존

- 환경변수: `HUGGINGFACE_TOKEN_READ` (모델 다운로드)
- `BACKEND = "cpu"` (현재 CPU. GPU 트랙에서 교체)
