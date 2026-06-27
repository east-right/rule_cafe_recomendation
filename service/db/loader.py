"""
CSV 데이터를 SQLite DB에 적재하는 스크립트
Usage:
  python loader.py            # cafe_keywords만 v2 재적재 (cafes/menus/review_summary 보존)
  python loader.py --full     # cafes까지 전체 재적재 (review_summary 초기화됨)

v2 변경:
- market_item.csv v2 컬럼(사업장명/category/키워드/대표descriptor/sentiment) 사용
- generic 키워드(generic_keywords.json)는 "키워드_descriptor" 형식으로 적재 → Soar operator와
  의미가 갈리는 키워드를 구분 매칭 (예: 매장_넓다 vs 매장_깔끔하다)
- single 키워드는 키워드만 그대로
- 같은 (매장, 키워드, sentiment) 중복 제거
"""

import csv
import json
import sys
from pathlib import Path

from schema import get_connection, init_db

ROOT = Path(__file__).resolve().parents[2]
CAFES_CSV = ROOT / "data" / "seoul_restaurants_shinline.csv"
KEYWORDS_CSV = ROOT / "data" / "market_item.csv"
MENU_CSV = ROOT / "data" / "menu_item.csv"
GENERIC_PATH = ROOT / "data" / "generic_keywords.json"


def load_cafes() -> None:
    # item이 있는 매장만 적재 (market_item.csv 기준)
    with open(KEYWORDS_CSV, encoding="utf-8-sig") as f:
        item_stores = {row["사업장명"] for row in csv.DictReader(f) if row["사업장명"]}

    # 주소/영업상태는 restaurant 원본에서 조회
    store_info: dict[str, tuple[str, str]] = {}
    with open(CAFES_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["사업장명"] in item_stores:
                store_info[row["사업장명"]] = (row["도로명주소"], row["상세영업상태명"])

    rows = [
        (name, info[0], info[1])
        for name, info in store_info.items()
    ]

    # item은 있지만 restaurant 목록에 없는 매장은 주소 없이 적재
    missing = item_stores - store_info.keys()
    for name in missing:
        rows.append((name, None, None))

    with get_connection() as conn:
        conn.execute("DELETE FROM cafe_keywords")
        conn.execute("DELETE FROM cafes")
        conn.executemany(
            "INSERT INTO cafes (name, address, status) VALUES (?, ?, ?)",
            rows,
        )
    print(f"[INFO] cafes 적재 완료: {len(rows)}개 (주소 없는 매장: {len(missing)}개)")


def load_keywords() -> None:
    generic = set(json.load(open(GENERIC_PATH, encoding="utf-8"))["generic"])

    seen: set[tuple] = set()
    rows: list[tuple] = []
    with open(KEYWORDS_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cafe = row["사업장명"]
            kw = row["키워드"]
            if not cafe or not kw:
                continue
            desc = row["대표descriptor"].strip()
            # generic 키워드 + descriptor 있으면 "키워드_descriptor", 아니면 키워드만
            keyword = f"{kw}_{desc}" if (kw in generic and desc) else kw
            key = (cafe, keyword, row["sentiment"])
            if key in seen:
                continue
            seen.add(key)
            rows.append((cafe, keyword, row["category"], row["sentiment"]))

    with get_connection() as conn:
        conn.execute("DELETE FROM cafe_keywords")
        conn.executemany(
            """INSERT INTO cafe_keywords (cafe_name, keyword, keyword_type, sentiment)
               VALUES (?, ?, ?, ?)""",
            rows,
        )
    n_generic = sum(1 for r in rows if "_" in r[1])
    print(f"[INFO] cafe_keywords 적재 완료: {len(rows)}개 (generic 분리형 {n_generic}개)")


def load_menus() -> None:
    """menu_item.csv(v2) → menus(유니크 메뉴명) + cafe_menus(매장별).
    약점여부 'Y'면 부정, 아니면 긍정. 같은 (매장,메뉴)는 부정 우선으로 유니크."""
    with open(MENU_CSV, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["사업장명"] and r["specific"]]

    # menus: 유니크 메뉴명
    unique_menus = sorted({r["specific"] for r in rows})

    # cafe_menus: (매장, 메뉴) 유니크, 약점여부 Y면 부정
    cafe_menu: dict[tuple, str] = {}
    for r in rows:
        key = (r["사업장명"], r["specific"])
        senti = "부정" if r["약점여부"].strip() == "Y" else "긍정"
        if key not in cafe_menu or senti == "부정":
            cafe_menu[key] = senti

    with get_connection() as conn:
        conn.execute("DELETE FROM cafe_menus")
        conn.execute("DELETE FROM menus")
        conn.executemany("INSERT INTO menus (name) VALUES (?)", [(m,) for m in unique_menus])
        conn.executemany(
            "INSERT INTO cafe_menus (cafe_name, menu_name, sentiment) VALUES (?, ?, ?)",
            [(c, m, s) for (c, m), s in cafe_menu.items()],
        )
    print(f"[INFO] menus 적재: {len(unique_menus)}개 (유니크) | cafe_menus 적재: {len(cafe_menu)}개")


if __name__ == "__main__":
    init_db()
    if "--full" in sys.argv:
        # 전체 재적재 (review_summary 초기화됨 — 주의)
        with get_connection() as conn:
            conn.execute("DELETE FROM cafe_keywords")
            conn.execute("DELETE FROM cafes")
        load_cafes()
        load_keywords()
        load_menus()
        print("[INFO] 전체 적재 완료")
    else:
        # 기본: cafe_keywords + 메뉴 v2 재적재 (cafes/review_summary 보존)
        load_keywords()
        load_menus()
        print("[INFO] cafe_keywords + 메뉴 v2 재적재 완료 (cafes/review_summary 보존)")
