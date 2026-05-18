import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "research.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id        TEXT UNIQUE NOT NULL,
            title           TEXT NOT NULL,
            abstract        TEXT,
            authors         TEXT,
            categories      TEXT,
            published_date  TEXT,
            pdf_link        TEXT,
            comments        TEXT,
            has_code        INTEGER DEFAULT 0,
            subfields       TEXT,
            difficulty_score REAL,
            difficulty_level TEXT
        );

        CREATE TABLE IF NOT EXISTS gaps (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            method              TEXT NOT NULL,
            domain              TEXT NOT NULL,
            paper_count         INTEGER DEFAULT 0,
            feasibility_score   REAL,
            feasibility_label   TEXT,
            status              TEXT,
            flags               TEXT,
            UNIQUE(method, domain)
        );

        CREATE TABLE IF NOT EXISTS trends (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword     TEXT NOT NULL,
            period      TEXT NOT NULL,
            frequency   REAL,
            growth_rate REAL,
            UNIQUE(keyword, period)
        );
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")


if __name__ == "__main__":
    init_db()