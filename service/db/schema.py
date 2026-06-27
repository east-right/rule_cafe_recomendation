import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "cafe.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cafes (
                name        TEXT PRIMARY KEY,
                address     TEXT,
                status      TEXT,
                review_summary TEXT
            );

            CREATE TABLE IF NOT EXISTS cafe_keywords (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                cafe_name    TEXT NOT NULL,
                keyword      TEXT NOT NULL,
                keyword_type TEXT NOT NULL,
                sentiment    TEXT NOT NULL,
                FOREIGN KEY (cafe_name) REFERENCES cafes(name)
            );

            CREATE INDEX IF NOT EXISTS idx_cafe_keywords_name
                ON cafe_keywords(cafe_name);

            CREATE INDEX IF NOT EXISTS idx_cafe_keywords_keyword
                ON cafe_keywords(keyword);

            CREATE TABLE IF NOT EXISTS menus (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cafe_menus (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                cafe_name  TEXT NOT NULL,
                menu_name  TEXT NOT NULL,
                sentiment  TEXT NOT NULL,
                menu_score REAL NOT NULL DEFAULT 0,
                FOREIGN KEY (cafe_name) REFERENCES cafes(name)
            );

            CREATE INDEX IF NOT EXISTS idx_cafe_menus_name
                ON cafe_menus(cafe_name);

            CREATE INDEX IF NOT EXISTS idx_cafe_menus_menu
                ON cafe_menus(menu_name);
        """)
    print(f"[INFO] DB 초기화 완료: {DB_PATH}")


if __name__ == "__main__":
    init_db()
