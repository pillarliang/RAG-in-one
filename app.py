import logging
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from constants.prompts import CN_RAG_PROMPTS
from constants.type import RAGRequest
from model.llm import DMetaLLM
from recall.faiss_wrapper import FaissWrapper
from recall.pre_retrieval import PreRetrievalService
from recall.retrieval import RetrievalService


logging.basicConfig(format='%(asctime)s %(pathname)s line:%(lineno)d [%(levelname)s] %(message)s', level='INFO')
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for local dev
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query")
def get_rag_res(request: RAGRequest):
    logger.info("Start handle query")
    query, chunks = request.query, request.chunks
    try:
        retrieval_chunks = set()
        faiss_wrapper = FaissWrapper(text_chunks=chunks)

        # Rephrase.
        queries = PreRetrievalService.rephrase_sub_queries(query)
        for item in queries:
            res = RetrievalService.semantic_search(
                item, faiss_wrapper, top_k=2
            )
            retrieval_chunks.update(res)

        # HYDE
        hyde_query = PreRetrievalService.hype(query)
        res = RetrievalService.semantic_search(
            hyde_query, faiss_wrapper, top_k=2
        )
        retrieval_chunks.update(res)

        # Generation
        prompts = CN_RAG_PROMPTS.format(question=query, contexts=retrieval_chunks)
        llm = DMetaLLM()
        res = llm.get_response(prompts)
        return res

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Internal Server Error")

@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
