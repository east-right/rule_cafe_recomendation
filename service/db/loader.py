"""
CSV 데이터를 SQLite DB에 적재하는 스크립트
Usage: python service/db/loader.py
"""

import csv
from pathlib import Path

from schema import get_connection, init_db

ROOT = Path(__file__).resolve().parents[2]
CAFES_CSV = ROOT / "data" / "seoul_restaurants_shinline.csv"
KEYWORDS_CSV = ROOT / "data" / "market_item.csv"


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
    with open(KEYWORDS_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [
            (row["사업장명"], row["최종_키워드"], row["키워드타입"], row["sentiment"])
            for row in reader
            if row["사업장명"] and row["최종_키워드"]
        ]

    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO cafe_keywords (cafe_name, keyword, keyword_type, sentiment)
               VALUES (?, ?, ?, ?)""",
            rows,
        )
    print(f"[INFO] cafe_keywords 적재 완료: {len(rows)}개")


if __name__ == "__main__":
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM cafe_keywords")
        conn.execute("DELETE FROM cafes")
    load_cafes()
    load_keywords()
    print("[INFO] 전체 적재 완료")
