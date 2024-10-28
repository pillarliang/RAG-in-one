from collections import defaultdict
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import asyncio
from openai import AsyncOpenAI
import os
from enum import Enum
import logging
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewChunk(BaseModel):
    """评论块模型"""
    content: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_type: SentimentType
    topics: List[str] = Field(default_factory=list)
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    cluster_id: int = Field(default=-1)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


class ReviewMetadata(BaseModel):
    """评论元数据"""
    total_chunks: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    main_topics: List[str] = Field(default_factory=list)


class ReviewAnalyzer:
    def __init__(self, openai_api_key: str):
        """初始化分析器"""
        # 加载spaCy模型
        try:
            self.nlp = spacy.load("zh_core_web_sm")
        except OSError:
            # 如果模型未下载，先下载模型
            os.system("python -m spacy download zh_core_web_sm")
            self.nlp = spacy.load("zh_core_web_sm")

        # 初始化OpenAI客户端
        self.client = AsyncOpenAI(api_key=openai_api_key)

    async def analyze_sentiment(self, text: str) -> tuple[float, SentimentType]:
        """使用OpenAI进行情感分析"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个情感分析专家。请分析提供文本的情感，" +
                                                  "返回一个-1到1之间的分数（-1表示最消极，1表示最积极）和情感类型（positive/negative/neutral）。" +
                                                  "只返回数字和类型，用逗号分隔。"},
                    {"role": "user", "content": text}
                ],
                max_tokens=10
            )
            result = response.choices[0].message.content
            score, sentiment = result.strip().split(',')
            return float(score), SentimentType(sentiment.strip().lower())
        except Exception as e:
            logger.error(f"情感分析出错: {str(e)}")
            return 0.0, SentimentType.NEUTRAL

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """使用spaCy提取实体"""
        doc = self.nlp(text)
        entities = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
        return dict(entities)

    def extract_topics(self, text: str) -> List[str]:
        """提取主题词"""
        doc = self.nlp(text)
        # 使用词性标注和依存句法分析提取主题
        topics = []
        for token in doc:
            # 选择名词短语作为主题
            if token.pos_ in ['NOUN', 'PROPN'] and token.dep_ in ['nsubj', 'dobj']:
                topics.append(token.text)
        return list(set(topics))


class ReviewChunker:
    def __init__(self, analyzer: ReviewAnalyzer, min_cluster_size: int = 5):
        self.analyzer = analyzer
        self.min_cluster_size = min_cluster_size
        self.vectorizer = TfidfVectorizer()

    async def create_chunk(self, text: str) -> ReviewChunk:
        """创建单个评论块"""
        sentiment_score, sentiment_type = await self.analyzer.analyze_sentiment(text)
        entities = self.analyzer.extract_entities(text)
        topics = self.analyzer.extract_topics(text)

        return ReviewChunk(
            content=text,
            sentiment_score=sentiment_score,
            sentiment_type=sentiment_type,
            topics=topics,
            entities=entities
        )

    async def create_chunks(self, reviews: List[str]) -> List[ReviewChunk]:
        """批量创建评论块"""
        # 并发处理所有评论
        chunks = await asyncio.gather(
            *[self.create_chunk(review) for review in reviews]
        )

        # 如果评论数量足够，进行聚类
        if len(chunks) >= self.min_cluster_size:
            # 准备文本数据进行聚类
            texts = [chunk.content for chunk in chunks]
            vectors = self.vectorizer.fit_transform(texts)

            # 确定聚类数量
            num_clusters = max(len(texts) // self.min_cluster_size, 1)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(vectors)

            # 更新聚类ID
            for i, chunk in enumerate(chunks):
                chunk.cluster_id = int(cluster_labels[i])

        return chunks


class EcommerceRAG:
    def __init__(self, openai_api_key: str):
        self.analyzer = ReviewAnalyzer(openai_api_key)
        self.chunker = ReviewChunker(self.analyzer)
        self.knowledge_base = {
            'sentiment': defaultdict(list),
            'topics': defaultdict(list),
            'entities': defaultdict(list),
            'clusters': defaultdict(list)
        }
        self.metadata = ReviewMetadata()

    async def process_reviews(self, reviews: List[str]):
        """处理评论并更新知识库"""
        chunks = await self.chunker.create_chunks(reviews)

        # 更新知识库
        for chunk in chunks:
            # 按情感分类
            self.knowledge_base['sentiment'][chunk.sentiment_type].append(chunk)

            # 按主题分类
            for topic in chunk.topics:
                self.knowledge_base['topics'][topic].append(chunk)

            # 按实体分类
            for entity_type, entities in chunk.entities.items():
                for entity in entities:
                    self.knowledge_base['entities'][entity].append(chunk)

            # 按聚类分类
            self.knowledge_base['clusters'][chunk.cluster_id].append(chunk)

        # 更新元数据
        self.update_metadata(chunks)

    def update_metadata(self, chunks: List[ReviewChunk]):
        """更新元数据统计信息"""
        self.metadata.total_chunks += len(chunks)
        for chunk in chunks:
            if chunk.sentiment_type == SentimentType.POSITIVE:
                self.metadata.positive_count += 1
            elif chunk.sentiment_type == SentimentType.NEGATIVE:
                self.metadata.negative_count += 1
            else:
                self.metadata.neutral_count += 1

        # 更新主要主题
        all_topics = []
        for chunk in chunks:
            all_topics.extend(chunk.topics)
        self.metadata.main_topics = list(set(all_topics))

    def query(self,
              query_type: str,
              key: Optional[str] = None,
              sentiment: Optional[SentimentType] = None,
              limit: int = 10) -> List[ReviewChunk]:
        """查询评论信息"""
        results = []

        if query_type == 'sentiment' and sentiment:
            results = self.knowledge_base['sentiment'].get(sentiment, [])
        elif query_type == 'topic' and key:
            results = self.knowledge_base['topics'].get(key, [])
        elif query_type == 'entity' and key:
            results = self.knowledge_base['entities'].get(key, [])
        elif query_type == 'cluster' and key is not None:
            results = self.knowledge_base['clusters'].get(int(key), [])

        # 排序并限制返回数量
        results.sort(key=lambda x: abs(x.sentiment_score), reverse=True)
        return results[:limit]

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "total_reviews": self.metadata.total_chunks,
            "sentiment_distribution": {
                "positive": self.metadata.positive_count,
                "negative": self.metadata.negative_count,
                "neutral": self.metadata.neutral_count
            },
            "main_topics": self.metadata.main_topics
        }


# 使用示例
async def demo_review_analysis():
    # 使用环境变量获取API密钥
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("请设置OPENAI_API_KEY环境变量")

    # 示例评论数据
    reviews = [
        "这个商品质量特别好，做工精致，物流也很快，五星好评！",
        "客服态度很差，退货流程复杂，很不满意。",
        "性价比很高，但是物流有点慢，包装还可以。",
        "产品不错，但是感觉价格稍微有点贵。",
        "收到货后发现有破损，售后服务态度还不错，及时处理了。"
    ]

    # 初始化系统
    rag = EcommerceRAG(openai_api_key)

    # 处理评论
    await rag.process_reviews(reviews)

    return rag


if __name__ == "__main__":
    async def main():
        try:
            rag = await demo_review_analysis()

            # 示例查询
            print("\n积极评价示例：")
            positive_reviews = rag.query(
                query_type='sentiment',
                sentiment=SentimentType.POSITIVE
            )
            for review in positive_reviews[:2]:
                print(review.model_dump_json(indent=2))

            # 获取统计信息
            stats = rag.get_statistics()
            print("\n统计信息：")
            print(stats)

        except Exception as e:
            logger.error(f"运行出错: {str(e)}")


    asyncio.run(main())