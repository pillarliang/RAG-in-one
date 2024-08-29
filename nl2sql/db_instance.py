import json
import os
import threading
import logging

from constants.prompts import CREATE_SAMPLE_SQL_FROM_TABLE
from constants.type import GenerateSQLResponse, RDBType, GenerateSampleSQLResponse
from model.llm import DMetaLLM
from nl2sql.database.sql_factory import rdb_factory
from core.vector_database.faiss_wrapper import FaissWrapper
from utility.db_state_machine import NL2SQLStateMachine, NL2SQLState
from utility.decorators import db_singleton
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@db_singleton
class DBInstance:
    """NL2SQL class for multiple instances of databases."""
    _state_machines: Dict[tuple, NL2SQLStateMachine] = {}
    _lock = threading.Lock()

    def __init__(
            self,
            db_type: Optional[str] = None,
            db_name: Optional[str] = None,
            db_host: Optional[str] = None,
            db_port: Optional[str] = None,
            db_user: Optional[str] = None,
            db_password: Optional[str] = None,
    ):
        self.db_type = db_type or os.getenv("LOCAL_DB_TYPE")
        self.db_name = db_name or os.getenv("LOCAL_DB_NAME")
        self.db_host = db_host or os.getenv("LOCAL_DB_HOST")
        self.db_port = db_port or os.getenv("LOCAL_DB_PORT")
        self.db_user = db_user or os.getenv("LOCAL_DB_USER")
        self.db_password = db_password or os.getenv("LOCAL_DB_PASSWORD")

        self.db = rdb_factory(db_type=self.db_type, db_name=self.db_name)

        self.llm = DMetaLLM()  # init LLM model
        self.db_summary = self.get_db_summary()
        self.sql_example = self._get_sql_example_llm()
        self.summary_index = FaissWrapper(text_chunks=self.db_summary)
        self.sql_example_index = FaissWrapper(text_chunks=self.sql_example)

        # init state machine
        self.db_key = (self.db_type, self.db_name)
        if self.db_key not in self._state_machines:
            with self._lock:
                if self.db_key not in self._state_machines:
                    self._state_machines[self.db_key] = NL2SQLStateMachine(self)

    def get_db_summary(self):
        """Get database summary used for constructing prompts."""
        return self.db.get_db_summary()

    @property
    def sql_example_llm(self):
        return self.sql_example

    def _get_sql_example_llm(self):
        """Get sql example: ⚠️有多少个表就调多少次 LLM，应该收集整理有代表性的 SQL 查询示例来代理每次 LLM 生成"""

        table_info = self.db.get_table_info()
        table_info_list = table_info.split("\n\n\n")
        sample_sql = []

        for table in table_info_list:
            response = self.llm.get_structured_response(CREATE_SAMPLE_SQL_FROM_TABLE.format(table_info=table),
                                                        response_format=GenerateSampleSQLResponse)
            sample_sql.extend(response["sql_list"])

        return sample_sql

    def db_update(self):
        """Notify the state machine of a database update."""
        if self.db_key in self._state_machines and self._state_machines[self.db_key].db_state is not NL2SQLState.UPDATING:
            logger.info(f"Updating state machine for {self.db_key}")
            with self._lock:
                self._state_machines[self.db_key].on_notification()
        else:
            logger.error(f"No state machine found for {self.db_key}. Is the instance initialized?")


if __name__ == "__main__":
    # # basic use case
    # db_instance = DBInstance(db_type=RDBType.MySQL.value, db_name="classicmodels")
    # res = db_instance.get_db_summary()
    # print(res)
    #
    # # 模拟数据库更新通知
    # print("\nSimulating database update for instance:")
    # db_instance.db_update()
    # res2 = db_instance.get_db_summary()
    # print(res2)

    db_instance = DBInstance(db_type=RDBType.MySQL.value, db_name="classicmodels")
    res = db_instance.sql_example_llm
    print(res)
