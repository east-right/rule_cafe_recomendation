"""market_item.csv의 MENU 타입 데이터로 menus, cafe_menus 테이블 생성."""
import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "market_item.csv"
DB_PATH = ROOT / "data" / "cafe.db"

conn = sqlite3.connect(str(DB_PATH))

conn.execute("DROP TABLE IF EXISTS cafe_menus")
conn.execute("DROP TABLE IF EXISTS menus")

conn.execute("""
CREATE TABLE menus (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
)
""")

conn.execute("""
CREATE TABLE cafe_menus (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    cafe_name TEXT NOT NULL,
    menu_name TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    UNIQUE(cafe_name, menu_name, sentiment)
)
""")

menu_set = set()
cafe_menu_pairs = set()

with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["키워드타입"] != "MENU":
            continue
        sentiment = row["sentiment"].strip()
        if sentiment not in ("긍정", "부정"):
            continue
        cafe_name = row["사업장명"].strip()
        menu_name = row["최종_키워드"].strip().split("_")[0].strip()
        if menu_name:
            menu_set.add(menu_name)
            cafe_menu_pairs.add((cafe_name, menu_name, sentiment))

conn.executemany("INSERT OR IGNORE INTO menus (name) VALUES (?)", [(m,) for m in sorted(menu_set)])
conn.executemany(
    "INSERT OR IGNORE INTO cafe_menus (cafe_name, menu_name, sentiment) VALUES (?, ?, ?)",
    sorted(cafe_menu_pairs),
)

conn.commit()

menu_count = conn.execute("SELECT COUNT(*) FROM menus").fetchone()[0]
cafe_menu_count = conn.execute("SELECT COUNT(*) FROM cafe_menus").fetchone()[0]
print(f"menus: {menu_count}개")
print(f"cafe_menus: {cafe_menu_count}개")

conn.close()
