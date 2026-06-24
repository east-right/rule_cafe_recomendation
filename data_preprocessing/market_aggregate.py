"""
비메뉴(FACILITY/ATMOSPHERE/TARGET) 매장 item 집계 → data/market_item.csv
- entity 단위 집계 + 대표descriptor(metadata)
- 긍정: TF-IDF 로 매장별 변별력 top5
- 부정: count 로 매장별 약점 top3 (TARGET 제외, 동점 시 global_count)
"""

import json, csv, collections, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
V3   = DATA / "ner_result_v3.json"
IDS  = DATA / "shinline_cafe_ids_v2.csv"
OUT  = DATA / "market_item.csv"

CATS     = ["FACILITY", "ATMOSPHERE", "TARGET"]
NEG_CATS = ["FACILITY", "ATMOSPHERE"]   # TARGET 부정은 데이터 적어 제외
POS_TOPN = 5
NEG_TOPN = 3


def is_valid(x: str) -> bool:
    return bool(x) and x not in ("null", "none", "None", "")


def load_names() -> dict:
    name = {}
    with open(IDS, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name[str(row["place_id"])] = row["사업장명"]
    return name


def main():
    data  = json.load(open(V3, encoding="utf-8"))
    names = load_names()
    N     = len(data)

    # store_cnt[pid][cat][senti][entity] = count
    store_cnt  = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(collections.Counter)))
    # store_desc[pid][cat][senti][entity] = Counter(descriptor)
    store_desc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(collections.Counter))))

    for pid, ents in data.items():
        for e in ents:
            cat = e["category"]
            if cat not in CATS:
                continue
            ent = e["entity"]
            if not is_valid(ent):
                continue
            senti = e["sentiment"]
            store_cnt[pid][cat][senti][ent] += 1
            d = e.get("descriptor")
            if is_valid(d):
                store_desc[pid][cat][senti][ent][d] += 1

    # 긍정 df (entity 등장 매장 수), 부정 global_count
    df_pos     = collections.defaultdict(collections.Counter)
    global_neg = collections.defaultdict(collections.Counter)
    for pid in store_cnt:
        for cat in CATS:
            for ent in store_cnt[pid][cat]["긍정"]:
                df_pos[cat][ent] += 1
            for ent, c in store_cnt[pid][cat]["부정"].items():
                global_neg[cat][ent] += c

    def rep_desc(pid, cat, senti, ent):
        top = store_desc[pid][cat][senti][ent].most_common(1)
        return top[0][0] if top else ""

    rows = []
    for pid in store_cnt:
        nm = names.get(pid, pid)
        for cat in CATS:
            # 긍정 TF-IDF top5
            pos   = store_cnt[pid][cat]["긍정"]
            total = sum(pos.values())
            scored = []
            for ent, c in pos.items():
                tf  = c / total if total else 0
                idf = math.log(N / df_pos[cat][ent]) if df_pos[cat][ent] else 0
                scored.append((tf * idf, c, ent))
            scored.sort(reverse=True)
            # count>=2 우선, 슬롯 남으면 count==1로 채움 (단발성 노이즈 억제)
            strong = [s for s in scored if s[1] >= 2]
            weak   = [s for s in scored if s[1] == 1]
            picked = (strong + weak)[:POS_TOPN]
            for score, c, ent in picked:
                rows.append({
                    "place_id": pid, "사업장명": nm, "category": cat,
                    "키워드": ent, "sentiment": "긍정",
                    "대표descriptor": rep_desc(pid, cat, "긍정", ent),
                    "count": c, "tfidf_score": round(score, 5),
                    "global_count": df_pos[cat][ent],
                })

            # 부정 count top3 (TARGET 제외)
            if cat in NEG_CATS:
                neg  = store_cnt[pid][cat]["부정"]
                cand = []
                for ent, c in neg.items():
                    gc = global_neg[cat][ent]
                    if c == 1 and gc == 1:     # 단발성 + 매우 희귀 → 제외
                        continue
                    cand.append((c, gc, ent))
                cand.sort(reverse=True)        # count desc, 동점 시 global_count desc
                for c, gc, ent in cand[:NEG_TOPN]:
                    rows.append({
                        "place_id": pid, "사업장명": nm, "category": cat,
                        "키워드": ent, "sentiment": "부정",
                        "대표descriptor": rep_desc(pid, cat, "부정", ent),
                        "count": c, "tfidf_score": "",
                        "global_count": gc,
                    })

    cols = ["place_id", "사업장명", "category", "키워드", "sentiment",
            "대표descriptor", "count", "tfidf_score", "global_count"]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    pos_n = sum(1 for r in rows if r["sentiment"] == "긍정")
    neg_n = sum(1 for r in rows if r["sentiment"] == "부정")
    print(f"완료: {len(rows)}행 (긍정 {pos_n} / 부정 {neg_n}) / 매장 {len(store_cnt)}개 → {OUT}")


if __name__ == "__main__":
    main()
