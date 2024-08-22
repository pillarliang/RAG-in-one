"""
Author: pillar
Date: 2024-08-22
Description: RetrievalService class for searching text chunks.
"""

import logging
from typing import List
from recall.faiss_wrapper import FaissWrapper

logger = logging.getLogger(__name__)


class RetrievalService:
    @staticmethod
    def retrieval(query: str, semantic_index, method="semantic", top_k: int = 3):
        if method == "hybrid":
            return RetrievalService.hybrid_search(query)
        elif method == "sql":
            return RetrievalService.sql_search(query)
        elif method == "semantic":
            return RetrievalService.semantic_search(query, semantic_index, top_k)

    @staticmethod
    def hybrid_search(query: str):
        # TODO: Implement hybrid search
        return query

    @staticmethod
    def sql_search(query: str):
        # TODO: Implement SQL search
        return query

    @staticmethod
    def semantic_search(query: str, semantic_index, top_k: int) -> List[str]:
        return semantic_index.search_for_chunks(query, top_k)


if __name__ == "__main__":
    query = "清华大学"
    text_chunk = [
        "他来到了未来",
        "我来到北京清华大学",
        "小明硕士毕业于中国科学院计算所",
        "我爱北京天安门",
    ]

    faiss_wrapper = FaissWrapper(text_chunks=text_chunk)

    hybrid_res = RetrievalService.semantic_search(
        query, faiss_wrapper, top_k=2
    )
    print(hybrid_res)
