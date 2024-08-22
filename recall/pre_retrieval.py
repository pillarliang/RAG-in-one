import json

from constants.type import RephraseQueryResponse, HydeResponse
from model.llm import DMetaLLM

"""
⚠️⚠️⚠️这一部分的效果的准确性未经过定量分析，仅用于跑通 demo 用。实际用需要在实际的数据集上测评下哪些组合的效果更优。
"""


class PreRetrievalService:
    llm = DMetaLLM()

    @classmethod
    def rephrase_sub_queries(cls, query):
        prompts = f"请根据重新润色问题{query}以更好的适合搜索，如果{query} 是一个复杂复杂问题，请将这个复杂问题拆分成多个子问题。如果能拆成子问题，子问题数不能超过 5 个"
        response = cls.llm.get_structured_response(prompts, RephraseQueryResponse)

        return json.loads(response)["rephrased_query"]

    @classmethod
    def hype(cls, query):
        prompts = f"请根据提供的问题{query}给出一个假设性的答案。答案要简洁明了，不需要详细的解释。"
        response = cls.llm.get_structured_response(prompts, HydeResponse)
        return json.loads(response)["hyde"]


if __name__ == "__main__":
    query = "小明的工作是什么？"
    res = PreRetrievalService.rephrase_sub_queries(query)
    print(res)
    res = PreRetrievalService.hype(query)
    print(res)
