"""
MENU 3계층 평가 테이블 → data/menu_item.csv
- v2 원본(specific 보존)에서 수집, 비메뉴(entity_MENU null)는 제외
- LLM 분류: category(고정 6) + type(누적)
- 매장 x specific 집계: 총언급/긍정/부정/중립/대표descriptor
- 맛집_score = (긍정-부정) x IDF   [게이트: 긍정비율>=0.6, 긍정>=2]
- 약점여부 = 부정비율 우세
"""

import json, csv, re, sys, collections, math
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
client = OpenAI()

DATA    = ROOT / "data"
V2      = DATA / "ner_result_v2.json"
IDS     = DATA / "shinline_cafe_ids_v2.csv"
MAP_DIR = DATA / "norm_mapping"
OUT     = DATA / "menu_item.csv"
CLASS_PATH = MAP_DIR / "menu_class.json"

CHUNK      = 40
CATEGORIES = ["커피", "음료", "디저트", "베이커리", "술", "푸드"]


def rule_norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())


def load_names() -> dict:
    name = {}
    with open(IDS, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name[str(row["place_id"])] = row["사업장명"]
    return name


def collect_menu():
    """v2에서 MENU 레코드 수집. entity_MENU에서 null로 분류된 비메뉴는 제외."""
    v2 = json.load(open(V2, encoding="utf-8"))
    entity_menu = json.load(open(MAP_DIR / "entity_MENU.json", encoding="utf-8"))
    null_set = {k for k, v in entity_menu.items() if v is None}

    records = []  # (pid, specific, sentiment, descriptor)
    for pid, ents in v2.items():
        for e in ents:
            if e["category"] != "MENU":
                continue
            spec = rule_norm(e["entity"])
            if not spec or spec in null_set:
                continue
            records.append((pid, spec, e["sentiment"], e.get("descriptor")))
    return records


def classify_chunk(chunk: list, types: list) -> dict:
    type_str = json.dumps(types, ensure_ascii=False) if types else "(아직 없음)"
    prompt = f"""카페 메뉴를 분류하세요. 각 메뉴에 category와 type을 부여합니다.

[category] 반드시 다음 중 하나: {CATEGORIES}
- 커피: 아메리카노/라떼/에스프레소 등 커피 음료
- 음료: 커피 아닌 음료(에이드/스무디/티/주스/밀크티 등)
- 디저트: 케이크/마카롱/푸딩/빙수 등 디저트
- 베이커리: 빵/스콘/크로플/타르트 등 구움과자·빵류
- 술: 주류
- 푸드: 샌드위치/샐러드/파스타 등 식사류

[type] 메뉴 종류명(온도/사이즈/재료 수식어 제거). 기존 type이 맞으면 재사용.
예: 아이스아메리카노->아메리카노, 바닐라라떼->라떼, 딸기라떼->라떼,
    쑥치즈케이크->치즈케이크, 소금빵->빵, 스콘->스콘, 휘낭시에->휘낭시에
기존 type 목록: {type_str}

입력: {json.dumps(chunk, ensure_ascii=False)}
출력(JSON): {{"메뉴명": {{"category": "...", "type": "..."}}, ...}}  (모든 입력 포함)"""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def classify(specifics: list) -> dict:
    cls = json.load(open(CLASS_PATH, encoding="utf-8")) if CLASS_PATH.exists() else {}
    types = sorted(set(v["type"] for v in cls.values() if v.get("type")))
    todo  = [s for s in specifics if s not in cls]

    for i in tqdm(range(0, len(todo), CHUNK), desc="menu_classify", file=sys.stdout):
        chunk = todo[i : i + CHUNK]
        res = classify_chunk(chunk, types)
        for k, v in res.items():
            if not isinstance(v, dict):
                continue
            cat = v.get("category")
            cls[k] = {
                "category": cat if cat in CATEGORIES else "푸드",
                "type": v.get("type") or k,
            }
            if cls[k]["type"] not in types:
                types.append(cls[k]["type"])
        with open(CLASS_PATH, "w", encoding="utf-8") as f:
            json.dump(cls, f, ensure_ascii=False, indent=2)
    return cls


def main():
    names   = load_names()
    records = collect_menu()
    specifics = sorted(set(r[1] for r in records))
    print(f"MENU 레코드 {len(records)}개 / unique specific {len(specifics)}개")

    print("=== category/type 분류 ===")
    cls = classify(specifics)

    # 매장 x specific 집계
    agg = collections.defaultdict(lambda: {"총": 0, "긍정": 0, "부정": 0, "중립": 0,
                                           "desc": collections.Counter()})
    for pid, spec, senti, desc in records:
        a = agg[(pid, spec)]
        a["총"] += 1
        if senti in ("긍정", "부정", "중립"):
            a[senti] += 1
        if desc and desc not in ("null", "none", "None", ""):
            a["desc"][desc] += 1

    # IDF: specific 을 파는 매장 수
    stores_per_spec = collections.defaultdict(set)
    for (pid, spec) in agg:
        stores_per_spec[spec].add(pid)
    N = len(set(pid for (pid, _) in agg))

    rows = []
    for (pid, spec), a in agg.items():
        pos, neg = a["긍정"], a["부정"]
        df = len(stores_per_spec[spec])
        idf = math.log(N / df) if df else 0

        ratio = pos / (pos + neg) if (pos + neg) else 0
        if pos >= 2 and ratio >= 0.6:
            matjip = round((pos - neg) * idf, 4)
        else:
            matjip = 0
        weak = (neg >= 2 and ratio < 0.4)

        c = cls.get(spec, {"category": "푸드", "type": spec})
        dd = a["desc"].most_common(1)
        rows.append({
            "place_id": pid, "사업장명": names.get(pid, pid),
            "category": c["category"], "type": c["type"], "specific": spec,
            "총언급": a["총"], "긍정": pos, "부정": neg, "중립": a["중립"],
            "대표descriptor": dd[0][0] if dd else "",
            "맛집score": matjip, "약점여부": "Y" if weak else "",
        })

    rows.sort(key=lambda r: (r["place_id"], -r["맛집score"], -r["총언급"]))

    cols = ["place_id", "사업장명", "category", "type", "specific",
            "총언급", "긍정", "부정", "중립", "대표descriptor", "맛집score", "약점여부"]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    matjip_n = sum(1 for r in rows if r["맛집score"] > 0)
    weak_n   = sum(1 for r in rows if r["약점여부"])
    print(f"완료: {len(rows)}행 / 맛집 {matjip_n} / 약점 {weak_n} / 매장 {N}개 → {OUT}")


if __name__ == "__main__":
    main()
