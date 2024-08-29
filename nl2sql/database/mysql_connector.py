from typing import Optional, Any, Dict, List, Type
from urllib.parse import quote
from urllib.parse import quote_plus as urlquote
import sqlalchemy
from sqlalchemy import MetaData, Table, create_engine, inspect, select, text

import logging
from nl2sql.database.sql_database import SQLDatabase

logger = logging.getLogger(__name__)


class MySQLConnector(SQLDatabase):
    """MySQL connector."""

    db_type: str = "mysql"
    driver: str = "mysql+pymysql"
    port: int = 3306


if __name__ == "__main__":
    db_user = "root"
    db_password = ""
    db_host = "localhost"
    db_name = "classicmodels"

    db = MySQLConnector.from_uri_db(
        host=db_host,
        user=db_user,
        password=db_password,
        db_name=db_name,
    )

    print(db.dialect)
    print(db.get_usable_table_names())
    print(f"table_info: {db.table_info}")

    print(db.get_indexes("customers"))
    print("=====")
    print(db.get_db_summary())
