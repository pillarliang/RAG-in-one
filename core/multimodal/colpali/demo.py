import os
from byaldi import RAGMultiModalModel

os.environ["HF_TOKEN"] = "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"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
RAG.index(
    input_path="docs/test.pdf",
    index_name="test",
    store_collection_with_index=False,
    overwrite = True,
)

query = "原始日志经过处理机器学习样本后还有什么操作？"
results = RAG.search(query, k=2)
print(results)
