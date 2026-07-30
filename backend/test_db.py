import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'ticketnow.db')

try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    print("SQLite Connected Successfully ✅")
    print(f"Database: {DB_PATH}")
    print(f"Tables: {[t[0] for t in tables]}")
except Exception as e:
    print("Connection Failed ❌")
    print(e)
