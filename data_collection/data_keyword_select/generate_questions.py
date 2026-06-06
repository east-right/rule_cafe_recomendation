"""
카페 추천 시스템 - 유효 질문 생성 스크립트 (Batch API 버전)
market_item.csv 키워드 기반으로 GPT Batch API로 질문을 생성합니다.
목표: 3000개 유효 질문 생성
"""

import pandas as pd
import json
import time
import random
import os
from openai import OpenAI
from pathlib import Path
from prompt import QUESTION_GENERATION_SYSTEM_PROMPT

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parents[2]
MARKET_ITEM_PATH  = BASE_DIR / "data" / "market_item.csv"
OUTPUT_PATH       = BASE_DIR / "data" / "valid_questions.json"
BATCH_INPUT_PATH  = BASE_DIR / "data" / "batch_input.jsonl"
BATCH_ID_PATH     = BASE_DIR / "data" / "batch_id.txt"
BATCH_OUTPUT_PATH = BASE_DIR / "data" / "batch_output.jsonl"

# ── 생성 설정 ──────────────────────────────────────────────────
TARGET_COUNT          = 3000  # 목표 질문 수
QUESTIONS_PER_PROMPT  = 5     # 프롬프트당 질문 수 (올림)
MAX_KEYWORDS_PER_TYPE = 50    # 타입별 키워드 샘플링 수 (올림)
COMBO_COUNT           = 80    # 복합 조합 수 (올림)
POLL_INTERVAL         = 60
# ──────────────────────────────────────────────────────────────


# ── 데이터 로드 ───────────────────────────────────────────────
def load_keywords(path: str) -> dict:
    df = pd.read_csv(path)
    keywords = {}
    for ktype in ["ATMOSPHERE", "FACILITY", "TARGET", "MENU"]:
        pos = (df[(df["키워드타입"] == ktype) & (df["sentiment"] == "긍정")]
               ["최종_키워드"].dropna().unique().tolist())
        neg = (df[(df["키워드타입"] == ktype) & (df["sentiment"] == "부정")]
               ["최종_키워드"].dropna().unique().tolist())
        keywords[ktype] = {"긍정": pos, "부정": neg}
    return keywords


def sample_kw(keywords: dict, ktype: str, n: int = 1, sentiment: str = "긍정") -> list:
    pool = keywords.get(ktype, {}).get(sentiment, [])
    return random.sample(pool, min(n, len(pool)))


# ── 유저 프롬프트 빌더 ────────────────────────────────────────
def user_prompt_single(ktype: str, kw: str, n: int) -> str:
    return f"Category: {ktype}\nKeyword: {kw}\nGenerate {n} Korean questions."

def user_prompt_multi(pairs: list, n: int) -> str:
    conds = "\n".join(f"- [{k}] {v}" for k, v in pairs)
    return f"Multi-condition:\n{conds}\nGenerate {n} Korean questions satisfying all conditions."


# ── 배치 요청 생성 ────────────────────────────────────────────
def build_batch_requests(keywords: dict) -> list:
    requests = []

    def make_req(custom_id: str, user_msg: str, meta: dict) -> dict:
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1-mini",
                "messages": [
                    {"role": "system", "content": QUESTION_GENERATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg}
                ],
                "temperature": 0.9,
                "max_tokens": 600,
            },
            "_meta": meta
        }

    # 1. 단일 타입 (ATMOSPHERE / FACILITY / TARGET)
    for ktype in ["ATMOSPHERE", "FACILITY", "TARGET"]:
        for i, kw in enumerate(sample_kw(keywords, ktype, n=MAX_KEYWORDS_PER_TYPE)):
            cid = f"single_{ktype}_{i}"
            req = make_req(
                cid,
                user_prompt_single(ktype, kw, QUESTIONS_PER_PROMPT),
                {"type": "single", "keywords": [{"type": ktype, "keyword": kw}]}
            )
            requests.append(req)

    # 2. MENU
    for i, kw in enumerate(sample_kw(keywords, "MENU", n=MAX_KEYWORDS_PER_TYPE)):
        cid = f"menu_{i}"
        req = make_req(
            cid,
            user_prompt_single("MENU", kw, QUESTIONS_PER_PROMPT),
            {"type": "menu", "keywords": [{"type": "MENU", "keyword": kw}]}
        )
        requests.append(req)

    # 3. 복합
    type_combos = [
        ("ATMOSPHERE", "TARGET"),
        ("ATMOSPHERE", "FACILITY"),
        ("TARGET", "FACILITY"),
        ("ATMOSPHERE", "TARGET", "FACILITY"),
    ]
    for i in range(COMBO_COUNT):
        combo = random.choice(type_combos)
        pairs = []
        for k in combo:
            kw_list = sample_kw(keywords, k, 1)
            if kw_list:
                pairs.append((k, kw_list[0]))
        if len(pairs) < 2:
            continue
        cid = f"multi_{i}"
        req = make_req(
            cid,
            user_prompt_multi(pairs, QUESTIONS_PER_PROMPT),
            {"type": "multi", "keywords": [{"type": k, "keyword": v} for k, v in pairs]}
        )
        requests.append(req)

    return requests


# ── 배치 제출 ─────────────────────────────────────────────────
def submit_batch(requests: list) -> str:
    Path(BATCH_INPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(BATCH_INPUT_PATH, "w", encoding="utf-8") as f:
        for req in requests:
            clean = {k: v for k, v in req.items() if k != "_meta"}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    meta_path = str(BATCH_INPUT_PATH).replace(".jsonl", "_meta.json")
    meta_map = {r["custom_id"]: r["_meta"] for r in requests}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_map, f, ensure_ascii=False, indent=2)

    print(f"📤 배치 파일 업로드 중... ({len(requests)}개 요청)")
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


# ── 배치 완료 대기 (while 폴링) ───────────────────────────────
def wait_for_batch(batch_id: str) -> str:
    print(f"⏳ 배치 완료 대기 중... (매 {POLL_INTERVAL}초 확인)")
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

        time.sleep(POLL_INTERVAL)


# ── 결과 파싱 ─────────────────────────────────────────────────
def parse_results(output_file_id: str) -> list:
    print("📥 결과 다운로드 중...")
    content = client.files.content(output_file_id).text

    with open(BATCH_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    meta_path = str(BATCH_INPUT_PATH).replace(".jsonl", "_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta_map = json.load(f)

    all_questions = []
    for line in content.strip().split("\n"):
        if not line:
            continue
        row  = json.loads(line)
        cid  = row["custom_id"]
        meta = meta_map.get(cid, {})

        try:
            text = row["response"]["body"]["choices"][0]["message"]["content"].strip()
            try:
                questions = json.loads(text)
            except json.JSONDecodeError:
                # max_tokens 잘림 복구
                last = text.rfind('",')
                if last == -1:
                    last = text.rfind('"')
                recovered = text[:last + 1] + "]" if last != -1 else None
                if recovered:
                    try:
                        questions = json.loads(recovered)
                        print(f"  ♻️  잘린 JSON 복구 [{cid}]: {len(questions)}개")
                    except Exception:
                        print(f"  ⚠️ 파싱 실패 [{cid}]: 복구 불가")
                        continue
                else:
                    print(f"  ⚠️ 파싱 실패 [{cid}]: 복구 불가")
                    continue

            if not isinstance(questions, list):
                continue
            for q in questions:
                all_questions.append({
                    "question": q,
                    "type":     meta.get("type", "unknown"),
                    "keywords": meta.get("keywords", []),
                    "label":    1
                })
        except Exception as e:
            print(f"  ⚠️ 파싱 실패 [{cid}]: {e}")

    return all_questions


# ── 중간 파일 정리 ────────────────────────────────────────────
def cleanup():
    targets = [
        BATCH_INPUT_PATH,
        BATCH_OUTPUT_PATH,
        BATCH_ID_PATH,
        Path(str(BATCH_INPUT_PATH).replace(".jsonl", "_meta.json")),
    ]
    for p in targets:
        Path(p).unlink(missing_ok=True)
    print("🧹 중간 파일 정리 완료")


# ── 기존 결과 로드 ────────────────────────────────────────────
def load_existing(path: Path) -> list:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"📋 기존 질문 {len(existing)}개 로드")
        return existing
    return []


# ── 메인 ─────────────────────────────────────────────────────
def main():
    meta_path = str(BATCH_INPUT_PATH).replace(".jsonl", "_meta.json")

    # 기존 결과 로드 (이어 생성용)
    all_questions = load_existing(OUTPUT_PATH)
    print(f"🎯 목표: {TARGET_COUNT}개 | 현재: {len(all_questions)}개 | 남은: {TARGET_COUNT - len(all_questions)}개")

    while len(all_questions) < TARGET_COUNT:
        remaining = TARGET_COUNT - len(all_questions)
        print(f"\n--- 배치 시작 (남은 목표: {remaining}개) ---")

        # 이미 제출된 배치 재사용
        if Path(BATCH_ID_PATH).exists():
            with open(BATCH_ID_PATH) as f:
                batch_id = f.read().strip()
            print(f"🔄 기존 배치 재사용: {batch_id}")

            # 메타 파일 없으면 재생성
            if not Path(meta_path).exists():
                print("⚠️  메타 파일 없음 → 재생성 중...")
                keywords = load_keywords(MARKET_ITEM_PATH)
                requests = build_batch_requests(keywords)
                meta_map = {r["custom_id"]: r["_meta"] for r in requests}
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_map, f, ensure_ascii=False, indent=2)
                print(f"✅ 메타 재생성 완료")
        else:
            keywords = load_keywords(MARKET_ITEM_PATH)
            requests = build_batch_requests(keywords)
            print(f"📋 {len(requests)}개 요청 생성")
            batch_id = submit_batch(requests)

        # 완료 대기 및 파싱
        output_file_id = wait_for_batch(batch_id)
        new_questions  = parse_results(output_file_id)
        all_questions.extend(new_questions)

        # 중복 제거
        seen = set()
        deduped = []
        for q in all_questions:
            if q["question"] not in seen:
                seen.add(q["question"])
                deduped.append(q)
        all_questions = deduped

        print(f"  이번 배치: {len(new_questions)}개 | 누적 (중복제거): {len(all_questions)}개")

        # 중간 저장
        Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_questions, f, ensure_ascii=False, indent=2)

        # 중간 파일 정리
        cleanup()

        if len(all_questions) >= TARGET_COUNT:
            break

        print(f"  목표 미달 → 다음 배치 시작...")

    # 최종 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_questions[:TARGET_COUNT], f, ensure_ascii=False, indent=2)

    print(f"\n✅ 완료! 총 {len(all_questions[:TARGET_COUNT])}개 질문 → {OUTPUT_PATH}")

    from collections import Counter
    for t, c in Counter(q["type"] for q in all_questions[:TARGET_COUNT]).items():
        print(f"  {t}: {c}개")


if __name__ == "__main__":
    main()