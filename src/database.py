"""
src/database.py
───────────────
SQLAlchemy engine, session management, and bulk-insert helpers.
Auto-disables when no DB credentials are configured (demo/CSV mode).
"""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATABASE_URL, USE_DATABASE, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger(__name__)

_engine = None
_Session = None


def get_engine():
    global _engine
    if not USE_DATABASE:
        raise RuntimeError("Database mode is disabled (no DB_PASSWORD set). Running in CSV-only mode.")
    if _engine is None:
        try:
            from sqlalchemy import create_engine
            _engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5)
            log.info("Database engine created: %s", DATABASE_URL.split("@")[-1])
        except Exception as e:
            log.error("Failed to create DB engine: %s", e)
            raise
    return _engine


@contextmanager
def get_session() -> Generator:
    from sqlalchemy.orm import sessionmaker
    global _Session
    if _Session is None:
        _Session = sessionmaker(bind=get_engine())
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def bulk_insert_df(df: pd.DataFrame, table_name: str, if_exists: str = "append", chunksize: int = 10_000):
    """Bulk insert a DataFrame into a PostgreSQL table."""
    engine = get_engine()
    log.info("Inserting %d rows into %s …", len(df), table_name)
    df.to_sql(table_name, con=engine, if_exists=if_exists, index=False, chunksize=chunksize, method="multi")
    log.info("Insert complete.")


def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Execute a SELECT query and return results as DataFrame."""
    engine = get_engine()
    return pd.read_sql(sql, con=engine, params=params)


def db_available() -> bool:
    """Return True if the DB connection can be established."""
    if not USE_DATABASE:
        return False
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False
