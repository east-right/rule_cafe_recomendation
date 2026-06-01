"""
Rule 제목 그룹핑 및 병합 스크립트

1단계: LLM으로 335개 title을 의미 기반 그룹핑
2단계: 그룹핑 결과로 rule_metadata.json 병합
3단계: 최종 rule_metadata_merged.json 저장
"""

import json
import os
import time
from openai import OpenAI
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[2]
INPUT_PATH   = BASE_DIR / "data" / "rule_metadata.json"
OUTPUT_PATH  = BASE_DIR / "data" / "rule_metadata_merged.json"
GROUP_PATH   = BASE_DIR / "data" / "title_groups.json"
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at grouping semantically similar Korean cafe recommendation rule titles.

Given a list of Korean rule titles, group titles that have the same or very similar meaning into clusters.
Assign a single representative title (the most clear and concise one) to each cluster.

Rules:
1. Group titles that mean essentially the same thing (e.g. 편안함/편안한/편안 → 편안함)
2. Keep titles that have distinct meanings as separate groups
3. Representative title should be the most commonly used or clearest form
4. For each group, assign confidence: "확신" or "애매"
   - 확신: members clearly mean the same thing
   - 애매: members are related but could be argued as distinct (e.g. 공부 vs 작업 vs 집중)
5. Add a short Korean reason explaining the grouping decision
6. Output ONLY valid JSON, no other text

Output format:
{
  "groups": [
    {
      "representative": "대표 제목",
      "members": ["제목1", "제목2", ...],
      "confidence": "확신 or 애매",
      "reason": "그룹핑 이유 한 줄"
    },
    ...
  ]
}"""


def get_title_groups(titles: list) -> dict:
    """LLM으로 title 그룹핑"""

    # 335개를 한번에 처리 (짧은 텍스트라 토큰 부담 없음)
    titles_str = json.dumps(titles, ensure_ascii=False)

    user_prompt = f"""Group these {len(titles)} Korean cafe rule titles by semantic similarity:

{titles_str}

Group titles with same/similar meaning. Output JSON only."""

    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.1,
            max_completion_tokens=16000,
        )
        text = response.choices[0].message.content.strip()
        # ```json 제거
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"⚠️ 그룹핑 실패: {e}")
        return None


def build_title_map(groups: dict) -> dict:
    """title → representative 매핑 딕셔너리 생성"""
    title_map = {}
    for group in groups["groups"]:
        rep = group["representative"]
        for member in group["members"]:
            title_map[member] = rep
    return title_map


def merge_rules(rules: list, title_map: dict, group_data: dict) -> list:
    """
    title_map 기반으로 rule 병합
    같은 representative로 매핑되는 rule들을 하나로 합침
    대표 rule: keywords 합집합, merged_questions 전부 수집
    """
    groups = defaultdict(list)
    for rule in rules:
        rep = title_map.get(rule["title"], rule["title"])
        groups[rep].append(rule)

    # confidence/reason 맵 구성
    conf_map = {}
    reason_map = {}
    for group in group_data["groups"]:
        rep = group["representative"]
        conf_map[rep]   = group.get("confidence", "확신")
        reason_map[rep] = group.get("reason", "")

    merged = []
    for rep, group_rules in groups.items():
        # keywords 합집합 (중복 제거)
        all_keywords = []
        seen = set()
        for r in group_rules:
            for kw in r.get("keywords", []):
                if kw not in seen:
                    seen.add(kw)
                    all_keywords.append(kw)

        # merged_questions 전부 수집
        all_merged_q = []
        for r in group_rules:
            all_merged_q.extend(r.get("merged_questions", []))

        # 대표 rule은 그룹 중 첫번째
        primary = group_rules[0]

        merged.append({
            "title":            rep,
            "description":      primary["description"],
            "keywords":         all_keywords,
            "question":         primary["question"],
            "merged_questions": all_merged_q,
            "source_count":     len(group_rules),
            "confidence":       conf_map.get(rep, "확신"),
            "reason":           reason_map.get(rep, ""),
        })

    return merged


def main():
    print("📂 rule_metadata.json 로드 중...")
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    rules  = data["unique_rules"]
    titles = list(set(r["title"] for r in rules))
    print(f"  총 rule: {len(rules)}개 | unique title: {len(titles)}개")

    # 기존 그룹핑 결과 있으면 재사용
    if GROUP_PATH.exists():
        print(f"🔄 기존 그룹핑 결과 재사용: {GROUP_PATH}")
        with open(GROUP_PATH, encoding="utf-8") as f:
            groups = json.load(f)
    else:
        print("🤖 LLM으로 title 그룹핑 중...")
        groups = get_title_groups(titles)
        if not groups:
            print("❌ 그룹핑 실패")
            return

        with open(GROUP_PATH, "w", encoding="utf-8") as f:
            json.dump(groups, f, ensure_ascii=False, indent=2)
        print(f"✅ 그룹핑 완료: {len(groups['groups'])}개 그룹 → {GROUP_PATH}")

    # title → representative 매핑
    title_map = build_title_map(groups)
    print(f"  title 매핑: {len(title_map)}개")

    # rule 병합
    print("🔀 rule 병합 중...")
    merged_rules = merge_rules(rules, title_map, groups)
    print(f"  병합 전: {len(rules)}개 → 병합 후: {len(merged_rules)}개")

    # 저장
    output = {
        "unique_rules": merged_rules,
        "title_map":    title_map,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 완료! → {OUTPUT_PATH}")

    # 통계
    from collections import Counter
    src_counts = Counter(r["source_count"] for r in merged_rules)
    print("\n병합 통계 (source_count 분포):")
    for cnt, num in sorted(src_counts.items()):
        print(f"  {cnt}개 rule 병합: {num}개")


if __name__ == "__main__":
    main()