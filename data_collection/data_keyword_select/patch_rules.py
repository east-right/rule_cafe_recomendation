"""
rule_metadata_merged.json 후처리 patch 적용 스크립트.

rule_patch.json(delete / merge) 설정을 읽어:
- merge: 대표 title <- [멤버 title들]. 멤버의 keywords / negative_keywords /
         merged_questions(+멤버 자신의 question)을 대표 rule로 합치고 멤버 rule 제거.
- delete: 해당 title rule 통째 제거.

원본은 rule_metadata_merged_prepatch.json 으로 백업.
"""

import json
import shutil
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parents[2]
RULE_PATH = BASE_DIR / "data" / "rule_metadata_merged.json"
BACKUP    = BASE_DIR / "data" / "rule_metadata_merged_prepatch.json"
PATCH     = Path(__file__).resolve().parent / "rule_patch.json"


def dedup_extend(base: list, extra: list) -> list:
    """순서 보존 합집합."""
    seen = set(base)
    out = list(base)
    for x in extra:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main():
    data = json.load(open(RULE_PATH, encoding="utf-8"))
    rules = data["unique_rules"]
    patch = json.load(open(PATCH, encoding="utf-8"))

    delete_set = set(patch.get("delete", []))
    merge_map  = patch.get("merge", {})

    by_title = {r["title"]: r for r in rules}
    print(f"[INFO] 적용 전: {len(rules)}개 rule")

    # ── merge ──────────────────────────────────────────────
    absorbed = set()   # 제거될 멤버 title
    for rep, members in merge_map.items():
        if rep not in by_title:
            print(f"  ⚠️ 대표 title 없음, 스킵: {rep}")
            continue
        rep_rule = by_title[rep]
        for m in members:
            if m == rep:
                continue
            if m not in by_title:
                print(f"  ⚠️ 멤버 title 없음, 스킵: {m} (→{rep})")
                continue
            mr = by_title[m]
            rep_rule["keywords"] = dedup_extend(
                rep_rule.get("keywords", []), mr.get("keywords", []))
            rep_rule["negative_keywords"] = dedup_extend(
                rep_rule.get("negative_keywords", []), mr.get("negative_keywords", []))
            extra_q = list(mr.get("merged_questions", []))
            if mr.get("question"):
                extra_q.append(mr["question"])
            rep_rule["merged_questions"] = dedup_extend(
                rep_rule.get("merged_questions", []), extra_q)
            rep_rule["source_count"] = rep_rule.get("source_count", 1) + mr.get("source_count", 1)
            absorbed.add(m)

    # ── delete + absorbed 제거 ─────────────────────────────
    drop = delete_set | absorbed
    kept = [r for r in rules if r["title"] not in drop]

    missing_del = delete_set - {r["title"] for r in rules}
    if missing_del:
        print(f"  ⚠️ delete 대상에 없는 title: {sorted(missing_del)}")

    # ── title_map 갱신 (멤버 → 대표) ───────────────────────
    tmap = data.get("title_map", {})
    for rep, members in merge_map.items():
        for m in members:
            if m != rep:
                tmap[m] = rep
    data["title_map"] = tmap

    # ── 저장 ───────────────────────────────────────────────
    shutil.copy(RULE_PATH, BACKUP)
    data["unique_rules"] = kept
    with open(RULE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] merge 흡수: {len(absorbed)}개 | delete: {len(delete_set & {r['title'] for r in rules})}개")
    print(f"[INFO] 적용 후: {len(kept)}개 rule")
    print(f"[INFO] 백업: {BACKUP.name} | 저장: {RULE_PATH.name}")


if __name__ == "__main__":
    main()
