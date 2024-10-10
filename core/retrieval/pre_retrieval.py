from typing import List
from constants.prompts import DECOMPOSE_QUERY_FOR_SQL
from constants.type import RephraseQueryResponse, HydeResponse, DecomposeQueryResponse
from model.llm import LLM

"""
⚠️⚠️⚠️这一部分的效果的准确性未经过定量分析，仅用于跑通 demo 用。实际用需要在实际的数据集上测评下哪些组合的效果更优。
"""


class PreRetrievalService:
    llm = LLM()

    @classmethod
    def rephrase_sub_queries(cls, query: str) -> List[str]:
        prompts = f"请根据重新润色问题{query}以更好的适合搜索，如果{query} 是一个复杂复杂问题，请将这个复杂问题拆分成多个子问题。如果能拆成子问题，子问题数不能超过 5 个"
        response: RephraseQueryResponse = cls.llm.get_response(prompts, RephraseQueryResponse)

        return response.rephrased_query

    @classmethod
    def hyde(cls, query: str) -> str:
        prompts = f"请根据提供的问题{query}给出一个假设性的答案。"
        response: HydeResponse = cls.llm.get_response(prompts, HydeResponse)
        return response.hyde

    @classmethod
    def decompose_for_sql(cls, query: str) -> DecomposeQueryResponse:
        response: DecomposeQueryResponse = cls.llm.get_response(DECOMPOSE_QUERY_FOR_SQL.format(question=query), DecomposeQueryResponse)
        # convert dict to pydantic model
        return response


if __name__ == "__main__":
    query = "小明的工作是什么？"
    # res = PreRetrievalService.rephrase_sub_queries(query)
    # print(res)
    res = PreRetrievalService.hyde(query)
    print(res)
    # sql_res = PreRetrievalService.decompose_for_sql(query)
    # print(sql_res)
