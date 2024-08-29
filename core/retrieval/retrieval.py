"""
Author: pillar
Date: 2024-08-22
Description: RetrievalService class for searching text chunks.
"""

import logging
from typing import List
from core.vector_database.jina_clip_wrapper import JinaClipWrapper
from utility.tools import load_images_from_folder

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

    @staticmethod
    def multimodal_search(query: str, multimodal_index, top_k: int):
        return multimodal_index.search_for_multimodal(query, top_k)


if __name__ == "__main__":
    # query = "清华大学"
    # text_chunk = [
    #     "他来到了未来",
    #     "我来到北京清华大学",
    #     "小明硕士毕业于中国科学院计算所",
    #     "我爱北京天安门",
    # ]
    #
    # faiss_wrapper = FaissWrapper(text_chunks=text_chunk)
    #
    # hybrid_res = RetrievalService.semantic_search(
    #     query, faiss_wrapper, top_k=2
    # )
    # print(hybrid_res)

    # usage2:
    query = "在 retrieval 之后应该做什么？"

    texts = ['A pig', 'A red cat', "A red pig"]
    image_folder = '../model/image_dataset'
    images = load_images_from_folder(image_folder)

    jina_clip_wrapper = JinaClipWrapper(texts, images)
    res = RetrievalService.multimodal_search(
        query, jina_clip_wrapper, top_k=2
    )

    print(res)
