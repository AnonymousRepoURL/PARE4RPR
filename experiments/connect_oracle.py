"""Small Oracle connection helper used by the import scripts.

Credentials are read from environment variables or a hidden prompt and are
never stored in source code.
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from getpass import getpass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import cx_Oracle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORACLE_NETWORK_CONFIG = Path(__file__).with_name("oracle_network")


def load_project_environment(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load simple KEY=VALUE entries from the git-ignored project .env file."""
    if not path.exists():
        return
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env entry on line {line_number}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key.replace("_", "").isalnum() or not key[0].isalpha():
            raise ValueError(f"Invalid .env key on line {line_number}: {key!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_project_environment()
cx_Oracle.init_oracle_client(config_dir=str(ORACLE_NETWORK_CONFIG))


class OracleConnection:
    def __init__(
        self,
        username: str,
        password: str,
        host: str = "127.0.0.1",
        port: int = 1521,
        service_name: str = "ORCL",
    ) -> None:
        self.dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
        self.username = username
        self.password = password

    @contextmanager
    def connect(self) -> Iterator[cx_Oracle.Connection]:
        connection = cx_Oracle.connect(
            user=self.username,
            password=self.password,
            dsn=self.dsn,
            encoding="UTF-8",
        )
        try:
            yield connection
        finally:
            connection.close()

    def execute_query(
        self, query: str, params: Optional[Mapping[str, Any]] = None
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params or {})
                columns = [item[0] for item in cursor.description]
                return columns, cursor.fetchall()

    def execute_update(
        self, query: str, params: Optional[Mapping[str, Any]] = None
    ) -> int:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params or {})
                rowcount = cursor.rowcount
            connection.commit()
            return rowcount


def password_from_environment_or_prompt(variable: str, username: str) -> str:
    password = os.environ.get(variable)
    return password if password else getpass(f"Oracle password for {username}: ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a local Oracle connection")
    parser.add_argument(
        "--username", default=os.environ.get("ORACLE_ADMIN_USER", "system")
    )
    parser.add_argument("--host", default=os.environ.get("ORACLE_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("ORACLE_PORT", "1521"))
    )
    parser.add_argument(
        "--service-name", default=os.environ.get("ORACLE_SERVICE_NAME", "ORCL")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = password_from_environment_or_prompt(
        "ORACLE_ADMIN_PASSWORD", args.username
    )
    oracle = OracleConnection(
        args.username, password, args.host, args.port, args.service_name
    )
    columns, rows = oracle.execute_query("""
        SELECT SYS_CONTEXT('USERENV', 'SESSION_USER') AS SESSION_USER,
               SYS_CONTEXT('USERENV', 'CON_NAME') AS CONTAINER_NAME,
               BANNER_FULL
        FROM V$VERSION
        WHERE BANNER_FULL LIKE 'Oracle Database%'
        """)
    print(dict(zip(columns, rows[0])))


if __name__ == "__main__":
    main()
