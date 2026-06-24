"""
카페 추천 시스템 - Rule 메타데이터 생성 스크립트 (single 질문 전용)

흐름:
1. valid_questions.json에서 type=single 질문만 필터링
2. 배치 API로 title/description/keywords 생성
3. 후처리: keywords 겹침 기준으로 중복 rule 병합
4. 최종 unique rule 목록 저장

산출물: data/rule_metadata.json
"""

import pandas as pd
import json
import time
import os
from openai import OpenAI
from pathlib import Path
from prompt import RULE_GENERATION_SYSTEM_PROMPT
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parents[2]
MARKET_ITEM_PATH  = BASE_DIR / "data" / "market_item.csv"
VALID_Q_PATH      = BASE_DIR / "data" / "valid_questions.json"
OUTPUT_PATH       = BASE_DIR / "data" / "rule_metadata.json"
BATCH_INPUT_PATH  = BASE_DIR / "data" / "rule_batch_input.jsonl"
BATCH_ID_PATH     = BASE_DIR / "data" / "rule_batch_id.txt"
BATCH_OUTPUT_PATH = BASE_DIR / "data" / "rule_batch_output.jsonl"

# ── 중복 병합 설정 ─────────────────────────────────────────────
MERGE_THRESHOLD = 0.5   # keywords 겹침 비율 이상이면 같은 rule로 판단
# ──────────────────────────────────────────────────────────────


# ── 데이터 로드 ───────────────────────────────────────────────
def load_keyword_pool() -> dict:
    """MENU 제외 3개 타입 긍정/부정 키워드 로드 (v2: 컬럼 category/키워드)
    pool[ktype] = {"긍정": [...], "부정": [...]}"""
    df = pd.read_csv(MARKET_ITEM_PATH)
    pool = {}
    for ktype in ["ATMOSPHERE", "FACILITY", "TARGET"]:
        sub = df[df["category"] == ktype]
        pool[ktype] = {
            "긍정": sub[sub["sentiment"] == "긍정"]["키워드"].dropna().unique().tolist(),
            "부정": sub[sub["sentiment"] == "부정"]["키워드"].dropna().unique().tolist(),
        }
    return pool


def load_single_questions() -> list:
    """valid_questions.json에서 type=single만 필터링"""
    with open(VALID_Q_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    single = [q for q in questions if q.get("type") == "single"]
    print(f"  전체 질문: {len(questions)}개 → single: {len(single)}개")
    return single


def build_keyword_context(pool: dict) -> str:
    """ATMOSPHERE/FACILITY/TARGET 긍정/부정 키워드 컨텍스트 구성"""
    lines = []
    for ktype in ["ATMOSPHERE", "FACILITY", "TARGET"]:
        lines.append(f"[{ktype} 긍정]\n" + ", ".join(pool[ktype]["긍정"]))
        neg = pool[ktype]["부정"]
        if neg:
            lines.append(f"[{ktype} 부정]\n" + ", ".join(neg))
    return "\n\n".join(lines)


# ── 프롬프트 빌더 ─────────────────────────────────────────────
def build_user_prompt(question: str, seed_keywords: list, keyword_context: str) -> str:
    seed_str = "\n".join(
        f"- [{s['type']}] {s['keyword']}"
        for s in seed_keywords if s["type"] != "MENU"
    )
    return f"""Question: {question}

Seed keywords (must be included if applicable):
{seed_str if seed_str else "(none)"}

Available keywords (긍정 = 갖춰야 할 특징, 부정 = 피해야 할 특징):
{keyword_context}

Generate rule metadata JSON."""


# ── 배치 요청 생성 ────────────────────────────────────────────
def build_batch_requests(questions: list, keyword_context: str) -> list:
    requests = []
    for i, q in enumerate(questions):
        question      = q["question"]
        seed_keywords = q.get("keywords", [])
        user_prompt   = build_user_prompt(question, seed_keywords, keyword_context)

        requests.append({
            "custom_id": f"rule_{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1-mini",
                "messages": [
                    {"role": "system", "content": RULE_GENERATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500,
            },
            "_meta": {"question": question, "seed_keywords": seed_keywords}
        })
    return requests


# ── 배치 제출/대기/파싱 ───────────────────────────────────────
def submit_batch(requests: list) -> str:
    Path(BATCH_INPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    meta_map  = {r["custom_id"]: r["_meta"] for r in requests}
    meta_path = str(BATCH_INPUT_PATH).replace(".jsonl", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_map, f, ensure_ascii=False, indent=2)

    with open(BATCH_INPUT_PATH, "w", encoding="utf-8") as f:
        for req in requests:
            clean = {k: v for k, v in req.items() if k != "_meta"}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    print(f"📤 배치 업로드 중... ({len(requests)}개 요청)")
    with open(BATCH_INPUT_PATH, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    batch_id = batch.id
    print(f"✅ 배치 제출 완료: {batch_id}")
    with open(BATCH_ID_PATH, "w") as f:
        f.write(batch_id)
    return batch_id


def wait_for_batch(batch_id: str) -> str:
    print("⏳ 배치 완료 대기 중...")
    while True:
        batch     = client.batches.retrieve(batch_id)
        status    = batch.status
        completed = batch.request_counts.completed
        total     = batch.request_counts.total
        failed    = batch.request_counts.failed
        print(f"  상태: {status} | 완료: {completed}/{total} | 실패: {failed}")
        if status == "completed":
            print("✅ 배치 완료!")
            return batch.output_file_id
        elif status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"배치 실패: {status}")
        time.sleep(60)


def parse_batch_results(output_file_id: str) -> list:
    print("📥 결과 다운로드 중...")
    content = client.files.content(output_file_id).text

    with open(BATCH_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    meta_path = str(BATCH_INPUT_PATH).replace(".jsonl", "_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta_map = json.load(f)

    results = []
    for line in content.strip().split("\n"):
        if not line:
            continue
        row  = json.loads(line)
        cid  = row["custom_id"]
        meta = meta_map.get(cid, {})

        try:
            text   = row["response"]["body"]["choices"][0]["message"]["content"].strip()
            result = json.loads(text)
            results.append({
                "question":          meta.get("question", ""),
                "seed_keywords":     meta.get("seed_keywords", []),
                "title":             result.get("title", ""),
                "description":       result.get("description", ""),
                "keywords":          result.get("keywords", []),
                "negative_keywords": result.get("negative_keywords", []),
            })
        except Exception as e:
            print(f"  ⚠️ 파싱 실패 [{cid}]: {e}")

    return results


# ── 중복 rule 병합 ────────────────────────────────────────────
def overlap_ratio(kw_a: list, kw_b: list) -> float:
    """두 키워드 리스트의 겹침 비율 계산 (Jaccard 유사도)"""
    set_a, set_b = set(kw_a), set(kw_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def merge_duplicate_rules(results: list, threshold: float = MERGE_THRESHOLD) -> list:
    """
    keywords 겹침 비율이 threshold 이상인 rule을 병합
    대표 rule: 먼저 나온 것 유지, 나머지는 representative 필드로 연결
    """
    unique_rules = []   # 대표 rule 목록
    merged_map   = {}   # question → 대표 rule title 매핑

    for result in results:
        kw      = result["keywords"]
        matched = None

        for rep in unique_rules:
            if overlap_ratio(kw, rep["keywords"]) >= threshold:
                matched = rep
                break

        if matched:
            # 기존 rule에 병합
            merged_map[result["question"]] = matched["title"]
            matched.setdefault("merged_questions", []).append(result["question"])
        else:
            # 새 rule 등록
            unique_rules.append({**result, "merged_questions": []})

    print(f"  원본: {len(results)}개 → 병합 후: {len(unique_rules)}개 unique rule")
    return unique_rules, merged_map


def cleanup():
    for p in [BATCH_INPUT_PATH, BATCH_OUTPUT_PATH, BATCH_ID_PATH,
              Path(str(BATCH_INPUT_PATH).replace(".jsonl", "_meta.json"))]:
        Path(p).unlink(missing_ok=True)
    print("🧹 중간 파일 정리 완료")


# ── 메인 ─────────────────────────────────────────────────────
CHUNK_SIZE    = 200
PROGRESS_PATH = BASE_DIR / "data" / "rule_progress.json"


def save_progress(chunk_idx: int, all_results: list):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump({"chunk_idx": chunk_idx, "results": all_results},
                  f, ensure_ascii=False, indent=2)


def load_progress() -> tuple:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        print(f"🔄 진행 상황 복원: chunk {data['chunk_idx']}번까지 완료, {len(data['results'])}개 결과")
        return data["chunk_idx"], data["results"]
    return 0, []


# ── 메인 ─────────────────────────────────────────────────────
CHUNK_SIZE    = 200
PROGRESS_PATH = BASE_DIR / "data" / "rule_progress.json"


def save_progress(chunk_idx: int, all_results: list):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump({"chunk_idx": chunk_idx, "results": all_results},
                  f, ensure_ascii=False, indent=2)


def load_progress() -> tuple:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        print(f"🔄 진행 상황 복원: chunk {data['chunk_idx']}번까지 완료, {len(data['results'])}개 결과")
        return data["chunk_idx"], data["results"]
    return 0, []


def main():
    print("📂 데이터 로드 중...")
    pool = load_keyword_pool()
    for ktype, kws in pool.items():
        print(f"  {ktype}: {len(kws)}개")

    questions       = load_single_questions()
    keyword_context = build_keyword_context(pool)

    # ── 테스트: 10개만 실행 ──────────────────────────────────
    # 결과 확인 후 아래 두 줄 주석 처리하고 전체 실행
    # questions = questions[:10]
    # print(f"🧪 테스트 모드: {len(questions)}개 질문만 실행")
    # ── 테스트 끝 ────────────────────────────────────────────

    chunks = [questions[i:i+CHUNK_SIZE] for i in range(0, len(questions), CHUNK_SIZE)]
    print(f"📦 총 {len(questions)}개 → {len(chunks)}개 청크 (청크당 {CHUNK_SIZE}개)")

    start_idx, all_results = load_progress()

    for idx, chunk in enumerate(chunks):
        if idx < start_idx:
            continue

        print(f"\n--- 청크 {idx+1}/{len(chunks)} ({len(chunk)}개 요청) ---")
        requests       = build_batch_requests(chunk, keyword_context)
        batch_id       = submit_batch(requests)
        output_file_id = wait_for_batch(batch_id)
        results        = parse_batch_results(output_file_id)
        all_results.extend(results)
        print(f"  누적 결과: {len(all_results)}개")
        save_progress(idx + 1, all_results)
        cleanup()

    print(f"\n🔀 중복 병합 중... (전체 {len(all_results)}개)")
    unique_rules, merged_map = merge_duplicate_rules(all_results)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"unique_rules": unique_rules, "merged_map": merged_map},
                  f, ensure_ascii=False, indent=2)

    PROGRESS_PATH.unlink(missing_ok=True)
    print(f"\n✅ 완료! unique rule {len(unique_rules)}개 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()