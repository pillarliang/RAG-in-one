import logging
from typing import Optional, List

from constants.prompts import GENERATE_SQL_PROMPTS, SQL_QUERY_ANSWER_PROMPTS, GENERATE_SQL_PROMPTS_V2
from constants.type import GenerateSQLResponse, RDBType
from core.retrieval.pre_retrieval import PreRetrievalService
from model.llm import DMetaLLM
from nl2sql.db_instance import DBInstance


logger = logging.getLogger(__name__)


class NL2SQLWorkflow:
    def __init__(self, db_instance: DBInstance):
        self.db_instance = db_instance
        self.llm = DMetaLLM()  # init LLM model

    def get_related_table_summary(self, query: str, top_k: int = 5):
        """Get related chunks based on the query."""
        return self.db_instance.summary_index.search_for_chunks(query, top_k=top_k)

    def get_sql_query(self, query: str, table_summary: Optional[str] = None, similarity_sql: Optional[List[str]] = None):
        """Get SQL query from the given query."""
        table_summary = table_summary if table_summary else self.get_related_table_summary(query)

        if similarity_sql:
            sql_res = self.llm.get_structured_response(
                GENERATE_SQL_PROMPTS_V2.format(table_info=table_summary, input=query, similarity_sql=similarity_sql),
                response_format=GenerateSQLResponse,
            )
        else:
            sql_res = self.llm.get_structured_response(
                GENERATE_SQL_PROMPTS.format(table_info=table_summary, input=query),
                response_format=GenerateSQLResponse,
            )

        logging.info(f"sql_res:{sql_res}")

        return sql_res["sql_query"]

    def get_sql_result(self, query: str, sql_query: Optional[str] = None):
        """executing the sql query."""
        sql_query = sql_query if sql_query else self.get_sql_query(query)
        logging.info(f"sql_query:{sql_query}")
        return self.db_instance.db.run_no_throw(sql_query)

    def get_sql_query_similarity(self, query: str, top_k: int = 5) -> List[str]:
        return self.db_instance.sql_example_index.search_for_chunks(query, top_k)

    def get_response(self, query: str, sql_query: Optional[str] = None, sql_res: Optional[str] = None):
        """Get response based on the query."""
        response = PreRetrievalService.decompose_for_sql(query)
        logging.info(f"text_to_sql_query:{response.datasets_query}")
        logging.info(f"rag_for_query:{response.interpretation_query}")

        if not sql_query:
            sql_query = self.get_sql_query(response.datasets_query)  # 第一次生成 sql 不准确，检索相似的正确样本，再重新生成一次。

        sql_similarity = self.get_sql_query_similarity(sql_query)  # 检索相似的正确 sql 样本
        sql_query = self.get_sql_query(query=response.datasets_query, similarity_sql=sql_similarity)  # 二次生成sql
        print(sql_query)

        if not sql_res:
            sql_res = self.get_sql_result(response.datasets_query, sql_query)

        llm_res = self.llm.get_response(
            query=SQL_QUERY_ANSWER_PROMPTS.format(question=query, query=response.interpretation_query, result=sql_res))

        return llm_res


if __name__ == "__main__":
    # basic use case
    instance = DBInstance(db_type=RDBType.MySQL.value, db_name="classicmodels")
    query = "what is price of `1968 Ford Mustang`"
    service = NL2SQLWorkflow(instance)
    res = service.get_response("what is price of `1968 Ford Mustang`")
    print(res)
