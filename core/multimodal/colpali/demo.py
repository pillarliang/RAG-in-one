import os
from byaldi import RAGMultiModalModel

from model.llm import DMetaLLM
from utility.tools import save_base64_image_2_local

os.environ["HF_TOKEN"] = "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"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["OPENAI_API_KEY"] = "sk-Fr7Bl02uYf5jXnkl4190783cFc414c68A2Fa75B68064FcDc"
os.environ["OPENAI_BASE_URL"] = "https://aihubmix.com/v1"


RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
RAG.index(
    input_path="./docs",
    index_name="test",
    store_collection_with_index=True,
    overwrite=True,
)

query = "现有 OCR 存在什么问题？"
results = RAG.search(query, k=1)

save_base64_image_2_local(results[0]["base64"], "test2.jpg")


contexts = {"images": [results[0]["base64"]]}
llm = DMetaLLM()
res = llm.get_multimodal_response(query=query, contexts=contexts)
print(res)
