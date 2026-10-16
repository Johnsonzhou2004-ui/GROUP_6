"""Small helpers to run the named SQL queries in sql/02_queries.sql."""
import re
import sqlite3
from functools import lru_cache

import pandas as pd

from src.config import DB_PATH, SQL_DIR


@lru_cache(maxsize=1)
def _load_queries() -> dict[str, str]:
    text = (SQL_DIR / "02_queries.sql").read_text()
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    # parts = [preamble, name1, body1, name2, body2, ...]
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def query_names() -> list[str]:
    return list(_load_queries())


def run_query(name: str, db_path=DB_PATH) -> pd.DataFrame:
    sql = _load_queries()[name]
    con = sqlite3.connect(db_path)
    try:
        return pd.read_sql(sql, con)
    finally:
        con.close()


def read_sql(sql: str, params=(), db_path=DB_PATH) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        return pd.read_sql(sql, con, params=params)
    finally:
        con.close()
