import logging
from contextlib import asynccontextmanager
import uvicorn
import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import ZhipuAIEmbeddings
from constants.prompts import CN_RAG_PROMPTS
from constants.type import RAGRequest
from model.llm import LLM
from core.hybrid_search.vector_database.faiss_wrapper import FaissWrapper
from core.retrieval.pre_retrieval import PreRetrievalService
from core.retrieval.retrieval import RetrievalService
from router import chat, streaming
from router.chat import postgres_db_client, redis_client

logging.basicConfig(format='%(asctime)s %(pathname)s line:%(lineno)d [%(levelname)s] %(message)s', level='INFO')
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting up the application...")
    # Initialize other resources if needed
    try:
        yield
    finally:
        # Shutdown logic
        logger.info("Shutting down the application...")
        postgres_db_client.engine.dispose()
        redis_client.close()
        logger.info("Resources have been cleaned up.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(chat.router)
app.include_router(streaming.router)


@app.post("/query")
def get_rag_res(request: RAGRequest):
    logger.info("Start handle query")
    query, chunks = request.query, request.chunks
    try:
        retrieval_chunks = set()
        faiss_wrapper = FaissWrapper(text_chunks=chunks, embedding=ZhipuAIEmbeddings())

        # Rephrase.
        queries = PreRetrievalService.rephrase_sub_queries(query)
        for item in queries:
            res = RetrievalService.semantic_search(
                item, faiss_wrapper, top_k=2
            )
            retrieval_chunks.update(res)

        # HYDE
        hyde_query = PreRetrievalService.hyde(query)
        res = RetrievalService.semantic_search(
            hyde_query, faiss_wrapper, top_k=2
        )
        retrieval_chunks.update(res)

        # Generation
        prompts = CN_RAG_PROMPTS.format(question=query, contexts=retrieval_chunks)
        llm = LLM()
        res = llm.get_response(prompts)
        return res

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Internal Server Error")


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
    