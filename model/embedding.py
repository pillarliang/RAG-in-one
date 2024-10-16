from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import os

# TODO：Transform the following setting into .env file
os.environ["OPENAI_API_KEY"] = "a2f2ae6bc9684b3706263c5d0ecc8ee2.fxFJYD33ocyYN25D"
os.environ["OPENAI_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4/"

zhipuAIEmbeddings_model = ZhipuAIEmbeddings(model="embedding-3", api_key="a2f2ae6bc9684b3706263c5d0ecc8ee2.fxFJYD33ocyYN25D", base_url="https://open.bigmodel.cn/api/paas/v4/")
openaiEmbeddings_model = OpenAIEmbeddings()

default_embedding_model = zhipuAIEmbeddings_model
