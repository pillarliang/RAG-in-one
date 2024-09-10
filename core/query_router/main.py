from enum import Enum
from typing import List
from pydantic import BaseModel

from constants.prompts import QueryRoutingPrompts
from model.llm import DMetaLLM


class SingleSelection(BaseModel):
    router_name: str
    reason: str
    
    
class QueryRouterName(str, Enum):
    RDB = "Relational Database"
    TextVector = "Text vector database"


class QueryRouterDescription(str, Enum):
    RDB = "This option is suitable when you need to retrieve specific information from a structured data source, such as a database table. For example, finding the price and inventory details of a product.",
    TextVector = "This option is suitable when you need to retrieve information from unstructured text data, such as locating relevant paragraphs or sentences from technical articles or news reports.",


class QueryRouterInfo(BaseModel):
    name: QueryRouterName
    description: QueryRouterDescription


class QueryRouter:
    def __init__(self):
        self.query_routes: List[QueryRouterInfo] = []

    def register(self, router_info: QueryRouterInfo):
        self.query_routes.append(router_info)

    def query(self, query: str) -> SingleSelection:
        prompts = QueryRoutingPrompts.SINGLE_SELECT.format(query=query, router_list=self.query_routes)
        llm = DMetaLLM()
        return llm.get_structured_response(query=prompts, response_format=SingleSelection)

    @classmethod
    def create(cls, config: List[QueryRouterName]) -> 'QueryRouter':
        router = cls()
        for route_name in config:
            router_info = QueryRouterInfo(name=route_name.value, description=QueryRouterDescription[route_name.name].value)
            router.register(router_info)
        return router


if __name__ == "__main__":
    router_2 = QueryRouter.create([QueryRouterName.RDB, QueryRouterName.TextVector])
    print(router_2.query_routes)
    query = "What is the capital of China?"
    
    res_2 = router_2.query(query)
    print("Result with 2 routes:", res_2)
    