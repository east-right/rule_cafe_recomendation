"""
독립 multi rule의 빈 keywords를 member single의 keywords 합집합으로 채운다.

build_multi_rules.py가 만든 multi는 title/description만 있고 keywords=[] 상태.
Soar 단계(keyword_extraction)는 keywords에서 operator/tiebreak를 뽑으므로,
multi의 두 member(single)의 keywords·negative_keywords를 합쳐 채워준다.

입력/출력: data/rule_metadata_merged.json (in-place, 백업 생성)
Usage: python fill_multi_keywords.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULE_PATH = ROOT / "data" / "rule_metadata_merged.json"
BACKUP_PATH = ROOT / "data" / "rule_metadata_merged_premultikw.json"


def _union(*lists):
    out = []
    for lst in lists:
        for x in lst:
            if x not in out:
                out.append(x)
    return out


def main():
    data = json.load(open(RULE_PATH, encoding="utf-8"))
    rules = data["unique_rules"]

    kw_map = {r["title"]: r.get("keywords", []) for r in rules}
    neg_map = {r["title"]: r.get("negative_keywords", []) for r in rules}

    filled, missing = 0, []
    for r in rules:
        if r.get("confidence") != "multi" or r.get("keywords"):
            continue
        members = r.get("members", [])
        kws = _union(*[kw_map.get(m, []) for m in members])
        negs = _union(*[neg_map.get(m, []) for m in members])
        if not kws:
            missing.append(r["title"])
            continue
        r["keywords"] = kws
        r["negative_keywords"] = negs
        filled += 1

    json.dump(data, open(BACKUP_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)  # 안전용 사본
    json.dump(data, open(RULE_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[INFO] multi keywords 채움: {filled}개")
    if missing:
        print(f"[WARN] member keywords 없어 못 채운 multi: {missing}")
    # 검증
    empty = [r["title"] for r in rules if not r.get("keywords")]
    print(f"[INFO] keywords 비어있는 rule 남음: {len(empty)}개 {empty[:10]}")


if __name__ == "__main__":
    main()
