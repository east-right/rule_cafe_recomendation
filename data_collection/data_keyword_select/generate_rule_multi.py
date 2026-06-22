"""
카페 추천 시스템 - Multi Rule 생성 스크립트

single rule로 조합 가능한 multi 질문만 대상으로
- seed_keywords가 속한 single rule들의 keywords 합집합을 컨텍스트로 제공
- LLM이 title + description + keywords 선택 (generate_rule.py와 동일 방식)
"""

import json
import time
import os
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from prompt import RULE_GENERATION_SYSTEM_PROMPT

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parents[2]
VALID_Q_PATH      = BASE_DIR / "data" / "valid_questions.json"
RULE_PATH         = BASE_DIR / "data" / "rule_metadata_merged.json"
OUTPUT_PATH       = BASE_DIR / "data" / "rule_metadata_merged.json"
BATCH_INPUT_PATH  = BASE_DIR / "data" / "multi_rule_batch_input.jsonl"
BATCH_ID_PATH     = BASE_DIR / "data" / "multi_rule_batch_id.txt"
BATCH_OUTPUT_PATH = BASE_DIR / "data" / "multi_rule_batch_output.jsonl"
# ──────────────────────────────────────────────────────────────


# ── 데이터 로드 ───────────────────────────────────────────────
def load_data():
    with open(VALID_Q_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    with open(RULE_PATH, encoding="utf-8") as f:
        rule_data = json.load(f)
    return questions, rule_data


def get_single_keywords_set(rule_data: dict) -> set:
    result = set()
    for rule in rule_data["unique_rules"]:
        for kw in rule["keywords"]:
            result.add(kw)
    return result


def get_kw_to_rule_keywords(rule_data: dict) -> dict:
    """keyword → 해당 keyword가 속한 rule의 전체 keywords"""
    kw_to_rule_keywords = {}
    for rule in rule_data["unique_rules"]:
        for kw in rule["keywords"]:
            if kw not in kw_to_rule_keywords:
                kw_to_rule_keywords[kw] = set(rule["keywords"])
            else:
                kw_to_rule_keywords[kw].update(rule["keywords"])
    return kw_to_rule_keywords


def get_kw_to_rule_negatives(rule_data: dict) -> dict:
    """keyword → 해당 keyword가 속한 rule의 negative_keywords 합집합"""
    kw_to_neg = {}
    for rule in rule_data["unique_rules"]:
        negs = rule.get("negative_keywords", [])
        for kw in rule["keywords"]:
            kw_to_neg.setdefault(kw, set()).update(negs)
    return kw_to_neg


def filter_coverable_multi(questions: list, single_keywords: set) -> list:
    multi = [q for q in questions if q["type"] == "multi"]
    return [
        q for q in multi
        if all(s["keyword"] in single_keywords for s in q["keywords"])
    ]


def filter_multi_rules(rules: list) -> list:
    """multi rule 후처리: title 3글자 이하 / single title 중복 / multi끼리 title 중복 제거.
    single rule은 전부 유지."""
    singles = [x for x in rules if x.get("confidence") != "multi"]
    multis  = [x for x in rules if x.get("confidence") == "multi"]
    single_titles = {x["title"] for x in singles}
    kept, seen = [], set()
    for m in multis:
        t = m["title"]
        if len(t) <= 3 or t in single_titles or t in seen:
            continue
        seen.add(t)
        kept.append(m)
    return singles + kept


def build_combined_keywords(q: dict, kw_to_rule_keywords: dict) -> list:
    """seed_keywords가 속한 rule들의 keywords 합집합"""
    combined = set()
    for s in q["keywords"]:
        kw = s["keyword"]
        if kw in kw_to_rule_keywords:
            combined.update(kw_to_rule_keywords[kw])
    return list(combined)


def build_combined_negatives(q: dict, kw_to_rule_negatives: dict) -> list:
    """seed_keywords가 속한 rule들의 negative_keywords 합집합"""
    combined = set()
    for s in q["keywords"]:
        kw = s["keyword"]
        if kw in kw_to_rule_negatives:
            combined.update(kw_to_rule_negatives[kw])
    return list(combined)


# ── 유저 프롬프트 빌더 ────────────────────────────────────────
def build_user_prompt(question: str, seed_keywords: list,
                      combined_keywords: list, combined_negatives: list) -> str:
    seed_str = "\n".join(
        f"- [{s['type']}] {s['keyword']}" for s in seed_keywords
    )
    pos_context = ", ".join(combined_keywords)
    neg_context = ", ".join(combined_negatives) if combined_negatives else "(none)"
    return f"""Question: {question}

Seed keywords (must be included if applicable):
{seed_str}

Available keywords (긍정 = 갖춰야 할 특징, selected from related rules):
{pos_context}

Available negative keywords (부정 = 피해야 할 특징):
{neg_context}

Generate rule metadata JSON."""


# ── 배치 요청 생성 ────────────────────────────────────────────
def build_batch_requests(questions: list, kw_to_rule_keywords: dict,
                         kw_to_rule_negatives: dict) -> list:
    requests = []
    for i, q in enumerate(questions):
        combined_kws  = build_combined_keywords(q, kw_to_rule_keywords)
        combined_negs = build_combined_negatives(q, kw_to_rule_negatives)
        user_prompt   = build_user_prompt(q["question"], q["keywords"],
                                          combined_kws, combined_negs)

        requests.append({
            "custom_id": f"multi_{i}",
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
            "_meta": {
                "question":      q["question"],
                "seed_keywords": q["keywords"],
            }
        })
    return requests


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


def parse_results(output_file_id: str) -> list:
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
                "title":             result.get("title", ""),
                "description":       result.get("description", ""),
                "keywords":          result.get("keywords", []),
                "negative_keywords": result.get("negative_keywords", []),
                "question":          meta.get("question", ""),
                "seed_keywords":     meta.get("seed_keywords", []),
                "merged_questions":  [],
                "source_count":      1,
                "confidence":        "multi",
                "reason":            "multi 질문 기반 조합 rule",
            })
        except Exception as e:
            print(f"  ⚠️ 파싱 실패 [{cid}]: {e}")

    return results


def cleanup():
    for p in [BATCH_INPUT_PATH, BATCH_OUTPUT_PATH, BATCH_ID_PATH,
              Path(str(BATCH_INPUT_PATH).replace(".jsonl", "_meta.json"))]:
        Path(p).unlink(missing_ok=True)
    print("🧹 중간 파일 정리 완료")


# ── 메인 ─────────────────────────────────────────────────────
def main():
    print("📂 데이터 로드 중...")
    questions, rule_data = load_data()

    single_keywords      = get_single_keywords_set(rule_data)
    kw_to_rule_keywords  = get_kw_to_rule_keywords(rule_data)
    kw_to_rule_negatives = get_kw_to_rule_negatives(rule_data)

    coverable = filter_coverable_multi(questions, single_keywords)
    print(f"  조합 가능한 multi 질문: {len(coverable)}개")

    if Path(BATCH_ID_PATH).exists():
        with open(BATCH_ID_PATH) as f:
            batch_id = f.read().strip()
        print(f"🔄 기존 배치 재사용: {batch_id}")
    else:
        requests = build_batch_requests(coverable, kw_to_rule_keywords, kw_to_rule_negatives)
        print(f"📋 총 {len(requests)}개 요청 생성")
        batch_id = submit_batch(requests)

    output_file_id = wait_for_batch(batch_id)
    new_rules      = parse_results(output_file_id)

    # 기존 rule_data에 추가
    rule_data["unique_rules"].extend(new_rules)
    print(f"\n✅ multi rule {len(new_rules)}개 추가 (필터 전 전체 {len(rule_data['unique_rules'])}개)")

    # 후처리 필터: 3글자 이하 / single title 중복 / multi끼리 중복 제거
    rule_data["unique_rules"] = filter_multi_rules(rule_data["unique_rules"])
    print(f"   필터 후 전체 rule 수: {len(rule_data['unique_rules'])}개")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rule_data, f, ensure_ascii=False, indent=2)

    cleanup()
    print(f"✅ 저장 완료 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()