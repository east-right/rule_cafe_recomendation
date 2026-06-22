"""
OpenAI Batch API를 이용한 매장 리뷰 요약 스크립트
GPT-4.1 배치 (50% 할인) 사용

Flow:
  1. market_review_summary.json 읽기
  2. 배치 JSONL 생성 및 업로드
  3. 배치 생성 → 완료 대기
  4. 결과 파싱 → cafes.review_summary UPDATE
"""

import json
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from prompt import REVIEW_SUMMARY_SYSTEM_PROMPT

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ROOT = Path(__file__).resolve().parents[2]
REVIEW_JSON = ROOT / "data" / "market_review_summary.json"
BATCH_INPUT_JSONL = ROOT / "data" / "review_summary_batch_input.jsonl"
BATCH_OUTPUT_JSONL = ROOT / "data" / "review_summary_batch_output.jsonl"
BATCH_ID_FILE = ROOT / "data" / "review_summary_batch_id.txt"
DB_PATH = ROOT / "data" / "cafe.db"

MODEL = "gpt-4.1"


def build_batch_jsonl(review_data: dict) -> None:
    with open(BATCH_INPUT_JSONL, "w", encoding="utf-8") as f:
        for store_name, data in review_data.items():
            reviews_text = "\n".join(f"- {r}" for r in data["reviews"])
            user_content = f"매장명: {store_name}\n\n리뷰:\n{reviews_text}"
            request = {
                "custom_id": store_name,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": REVIEW_SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": 300,
                    "temperature": 0.3,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")


def submit_batch(client: OpenAI) -> str:
    print("[INFO] 파일 업로드 중...")
    with open(BATCH_INPUT_JSONL, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  파일 ID: {uploaded.id}")

    print("[INFO] 배치 생성 중...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  배치 ID: {batch.id}")
    BATCH_ID_FILE.write_text(batch.id, encoding="utf-8")
    return batch.id


def wait_for_batch(client: OpenAI, batch_id: str) -> str:
    print("[INFO] 배치 완료 대기 중...")
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  상태: {batch.status} | 완료: {counts.completed}/{counts.total}")
        if batch.status in terminal:
            if batch.status != "completed":
                raise RuntimeError(f"배치 실패: {batch.status}")
            return batch.output_file_id
        time.sleep(60)


def parse_results(client: OpenAI, output_file_id: str) -> dict[str, str]:
    print("[INFO] 결과 다운로드 중...")
    content = client.files.content(output_file_id)
    BATCH_OUTPUT_JSONL.write_bytes(content.content)

    results: dict[str, str] = {}
    errors = 0
    with open(BATCH_OUTPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            store = item["custom_id"]
            if item.get("error") is None:
                summary = item["response"]["body"]["choices"][0]["message"]["content"].strip()
                results[store] = summary
            else:
                print(f"  [WARN] 오류: {store} — {item['error']}")
                errors += 1

    print(f"  성공: {len(results)}개, 오류: {errors}개")
    return results


def update_db(results: dict[str, str]) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executemany(
            "UPDATE cafes SET review_summary = ? WHERE name = ?",
            [(summary, name) for name, summary in results.items()],
        )
        conn.commit()
        updated = conn.execute(
            "SELECT COUNT(*) FROM cafes WHERE review_summary IS NOT NULL"
        ).fetchone()[0]
        print(f"[INFO] DB 업데이트 완료: review_summary 보유 매장 {updated}개")
    finally:
        conn.close()


def main() -> None:
    client = OpenAI()

    print("[INFO] 리뷰 데이터 로드 중...")
    with open(REVIEW_JSON, encoding="utf-8") as f:
        review_data = json.load(f)
    print(f"  매장 수: {len(review_data)}")

    print("[INFO] 배치 JSONL 생성 중...")
    build_batch_jsonl(review_data)
    print(f"  저장: {BATCH_INPUT_JSONL}")

    batch_id = submit_batch(client)
    output_file_id = wait_for_batch(client, batch_id)
    results = parse_results(client, output_file_id)
    update_db(results)

    print("\n[INFO] 전체 완료")


if __name__ == "__main__":
    main()
