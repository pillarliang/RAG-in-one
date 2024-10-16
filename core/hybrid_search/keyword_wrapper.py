import logging
from typing import List, Tuple

import spacy
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class KeywordWrapper:
    def __init__(self, text_chunks, language="zh") -> None:
        self.language = language
        self.text_chunks = text_chunks
        tokenized_chunks = self.tokenize_sentences(self.text_chunks)
        self.index = BM25Okapi(tokenized_chunks)

    def tokenize_sentence(self, sentence: str):
        """
        Tokenize a sentence using spacy
        """
        nlp = spacy.load(f"{self.language}_core_web_sm")  # ⚠️⚠️⚠️ <python -m spacy download zh_core_web_sm> ⚠️⚠️⚠️
        docs = nlp(sentence)
        return [token.text for token in docs]

    def tokenize_sentences(self, sentences: List[str]):
        """
        Tokenize a list of sentences
        """
        return [self.tokenize_sentence(sentence) for sentence in sentences]

    def search_for_scores(self, query: str) -> List[float]:
        """
        Get the BM25 scores for a query
        """
        tokenized_query = self.tokenize_sentence(query)
        bm25_scores = self.index.get_scores(tokenized_query)
        return bm25_scores.tolist()

    def search_for_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """
        Get the top k text chunks for a query
        """
        scores = self.search_for_scores(query)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.text_chunks[i] for i in top_k_indices]

    def search_for_chunks_with_scores(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get the top k text chunks with their corresponding scores for a query
        """
        scores = self.search_for_scores(query)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.text_chunks[i], scores[i]) for i in top_k_indices]

    def destroy_index(self):
        """
        释放索引以释放内存。
        """
        del self.index
        self.index = None


if __name__ == "__main__":
    query = "清华大学"
    text_chunk = [
        "他来到了网易杭研大厦",
        "我来到北京清华大学",
        "小明硕士毕业于中国科学院计算所",
        "我爱北京天安门",
    ]
    keyword_wrapper = KeywordWrapper(text_chunk)
    res = keyword_wrapper.search_for_scores(query)
    chunks_res = keyword_wrapper.search_for_chunks(query)
    chunks_scores_res = keyword_wrapper.search_for_chunks_with_scores(query)
    print(res)
    print(chunks_res)
    print(chunks_scores_res)
