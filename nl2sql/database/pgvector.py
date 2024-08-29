import numpy as np
from sqlalchemy import create_engine, Table, Column, String, Text, MetaData, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import select
import uuid

# 创建数据库连接
connection_string = "postgresql+psycopg://liangzhu:@localhost:5432/nl2sql"
engine = create_engine(connection_string)
metadata = MetaData()

# 创建扩展（如果不存在）
with engine.connect() as connection:
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    connection.execute("CREATE EXTENSION IF NOT EXISTS hstore;")
    connection.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")

# 定义一个示例表
vector_store = Table(
    "vector_store",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("content", Text),
    Column("metadata", JSON),
    Column("embedding", String)  # 暂时使用String来模拟vector类型
)

# 创建表（如果不存在）
metadata.create_all(engine)

# 插入示例数据
with engine.connect() as connection:
    # 开始一个事务
    with connection.begin():
        # 插入示例向量数据
        connection.execute(
            vector_store.insert(),
            [
                {"content": "Example 1", "metadata": {}, "embedding": "{0.1,0.2,0.3}"},
                {"content": "Example 2", "metadata": {}, "embedding": "{0.4,0.5,0.6}"},
                {"content": "Example 3", "metadata": {}, "embedding": "{0.7,0.8,0.9}"},
            ]
        )

# 查询和计算向量相似度（例如：余弦相似度）
query_vector = np.array([0.1, 0.2, 0.3])

with engine.connect() as connection:
    # 选择所有数据，并计算与查询向量的余弦相似度
    result = connection.execute(
        select(
            vector_store.c.id, vector_store.c.embedding
        )
    )

    for row in result:
        # 将字符串转换为numpy数组
        db_vector = np.array(eval(row.embedding))
        similarity = np.dot(query_vector, db_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(db_vector))
        print(f"ID: {row.id}, Similarity: {similarity:.4f}")
