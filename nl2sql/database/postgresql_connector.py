from typing import Optional, Any
from urllib.parse import quote
from urllib.parse import quote_plus as urlquote
from langchain_community.utilities.sql_database import SQLDatabase
import logging

logger = logging.getLogger(__name__)


class PostgreSQLConnector(SQLDatabase):
    """PostgreSQL connector."""

    driver = "postgresql+psycopg"
    db_type = "postgresql"
    port = 5432

    @classmethod
    def from_uri_db(
        cls,
        host: str,
        user: str,
        pwd: str,
        db_name: str,
        port: int = 3306,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> SQLDatabase:
        """Construct a SQLAlchemy engine from uri database.

        Args:
            host (str): database host.
            port (int): database port.
            user (str): database user.
            pwd (str): database password.
            db_name (str): database name.
            engine_args (Optional[dict]):other engine_args.
        """
        db_url: str = (
            f"{cls.driver}://{quote(user)}:{urlquote(pwd)}@{host}:{str(port)}/{db_name}"
        )
        return cls.from_uri(db_url, engine_args, **kwargs)
