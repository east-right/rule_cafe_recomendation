"""OpenAI Batch API 재사용 하니스.

모든 GPT 호출(질문 생성, 리뷰 채점 등)을 이 배치 인터페이스로 돌린다.
- build_request: 요청 1건(jsonl 한 줄) 생성
- submit: 요청 리스트 → 배치 제출, batch_id 저장
- poll: 완료까지 대기(타임아웃 시 현재 상태 반환)
- fetch: 결과 jsonl → {custom_id: body} 딕셔너리

비싼 채점 단계도 이 함수들을 그대로 재사용한다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Windows 콘솔(cp949)에서 한글/em-dash 출력 시 크래시 방지
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _find_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / ".env").exists():
            return parent
    return start


HERE = Path(__file__).resolve().parent
ROOT = _find_root(HERE)
load_dotenv(ROOT / ".env")


def build_request(custom_id: str, messages: list[dict], model: str = "gpt-4.1-mini",
                  **body) -> dict:
    """배치 jsonl 한 줄(요청 1건)."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model, "messages": messages, **body},
    }


def submit(requests: list[dict], name: str, client: OpenAI | None = None) -> str:
    """요청 리스트를 배치로 제출하고 batch_id를 eval_data/<name>_batch_id.txt에 저장."""
    client = client or OpenAI()
    inp = HERE / f"{name}_input.jsonl"
    with inp.open("w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with inp.open("rb") as fh:
        file_obj = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    (HERE / f"{name}_batch_id.txt").write_text(batch.id, encoding="utf-8")
    print(f"[batch] 제출 완료: {name}  id={batch.id}  요청 {len(requests)}건")
    return batch.id


def poll(batch_id: str, client: OpenAI | None = None,
         interval: int = 15, timeout: int = 540):
    """완료/실패까지 대기. 타임아웃이면 마지막 상태 객체를 그대로 반환."""
    client = client or OpenAI()
    start = time.time()
    while True:
        b = client.batches.retrieve(batch_id)
        done = b.request_counts.completed if b.request_counts else 0
        total = b.request_counts.total if b.request_counts else 0
        print(f"[batch] status={b.status}  {done}/{total}")
        if b.status in ("completed", "failed", "expired", "cancelled"):
            return b
        if time.time() - start > timeout:
            print("[batch] 타임아웃 — 아직 진행 중. 나중에 다시 실행하면 이어받음.")
            return b
        time.sleep(interval)


def fetch(batch, name: str, client: OpenAI | None = None) -> dict[str, dict]:
    """완료된 배치의 결과를 {custom_id: response_body}로 반환 + 원본 jsonl 저장."""
    client = client or OpenAI()
    if batch.error_file_id:
        err = client.files.content(batch.error_file_id).text
        (HERE / f"{name}_errors.jsonl").write_text(err, encoding="utf-8")
        print(f"[batch] ⚠️ 에러 라인 있음 → {name}_errors.jsonl")
    text = client.files.content(batch.output_file_id).text
    (HERE / f"{name}_output.jsonl").write_text(text, encoding="utf-8")
    out: dict[str, dict] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        out[o["custom_id"]] = o["response"]["body"]
    return out
