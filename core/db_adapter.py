#!/usr/bin/env python3
"""
EIMAS DB Adapter
================
SQLite(개발) / PostgreSQL(운영) 환경 자동 전환 어댑터.

환경변수:
    EIMAS_DB_URL=sqlite:///data/eimas.db        # 개발 (기본값)
    EIMAS_DB_URL=postgresql://user:pw@host/db   # 운영

사용법:
    adapter = create_adapter()
    with adapter.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(adapter.adapt_sql("SELECT * WHERE id=?"), (1,))
"""

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

# 환경변수 이름
EIMAS_DB_URL = "EIMAS_DB_URL"

# 기본 SQLite 경로 (database.py와 동일 위치)
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "eimas.db"
DEFAULT_DB_URL = f"sqlite:///{_DEFAULT_DB_PATH}"


class DBAdapter(ABC):
    """DB 연결 추상 어댑터 — SQLite / PostgreSQL 공통 인터페이스."""

    # SQL 파라미터 플레이스홀더 (서브클래스에서 오버라이드)
    placeholder: str = "?"

    @property
    @abstractmethod
    def backend(self) -> str:
        """백엔드 이름 ('sqlite' | 'postgresql')"""

    @contextmanager
    @abstractmethod
    def connection(self) -> Iterator[Any]:
        """컨텍스트 매니저: 커밋/롤백/클로즈 자동 처리."""

    def adapt_sql(self, sql: str) -> str:
        """플레이스홀더 방언 변환 (필요 시 서브클래스 오버라이드)."""
        return sql


class SQLiteAdapter(DBAdapter):
    """SQLite 어댑터 — 개발/테스트 환경."""

    placeholder = "?"

    def __init__(self, db_path: str):
        import sqlite3
        self._db_path = str(db_path)
        self._sqlite3 = sqlite3
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    @property
    def backend(self) -> str:
        return "sqlite"

    @contextmanager
    def connection(self):
        conn = self._sqlite3.connect(self._db_path)
        conn.row_factory = self._sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


class PostgreSQLAdapter(DBAdapter):
    """PostgreSQL 어댑터 — 운영 환경.

    요구사항:
        pip install psycopg2-binary

    DDL 주의사항:
        - INTEGER PRIMARY KEY AUTOINCREMENT  →  SERIAL PRIMARY KEY
        - INSERT OR REPLACE  →  INSERT ... ON CONFLICT DO UPDATE SET ...
        - TEXT DEFAULT CURRENT_TIMESTAMP  →  TIMESTAMPTZ DEFAULT NOW()
    """

    placeholder = "%s"

    def __init__(self, dsn: str):
        self._dsn = dsn

    @property
    def backend(self) -> str:
        return "postgresql"

    @contextmanager
    def connection(self):
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise ImportError(
                "psycopg2가 설치되어 있지 않습니다. "
                "운영 환경에서 실행하려면: pip install psycopg2-binary"
            )
        conn = psycopg2.connect(self._dsn)
        conn.autocommit = False
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def adapt_sql(self, sql: str) -> str:
        """SQLite '?' 플레이스홀더를 PostgreSQL '%s'로 변환."""
        return sql.replace("?", "%s")


def create_adapter(db_url: str = None) -> DBAdapter:
    """
    DB_URL에 따라 SQLiteAdapter 또는 PostgreSQLAdapter 반환.

    Args:
        db_url: 명시적 URL. None이면 EIMAS_DB_URL 환경변수 → 기본 SQLite 순으로 적용.

    Examples:
        # 개발
        adapter = create_adapter("sqlite:///data/eimas.db")

        # 운영
        adapter = create_adapter("postgresql://eimas:secret@db-host:5432/eimas_prod")

        # 환경변수 기반 자동 전환
        os.environ["EIMAS_DB_URL"] = "postgresql://..."
        adapter = create_adapter()  # PostgreSQLAdapter 반환
    """
    url = db_url or os.getenv(EIMAS_DB_URL, DEFAULT_DB_URL)

    if url.startswith(("postgresql://", "postgres://")):
        return PostgreSQLAdapter(url)

    # sqlite:///path 또는 단순 파일 경로
    path = url.removeprefix("sqlite:///")
    return SQLiteAdapter(path)
