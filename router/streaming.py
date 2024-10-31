import logging
from typing import AsyncGenerator
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from model.llm import LLM
import streamingjson

from model.streaming import JSONStreamProcessor

router = APIRouter(prefix="/streaming", tags=["streaming"])
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str


class Topic(BaseModel):
    positive: str
    negative: str
    count: str = Field("2")


class Response(BaseModel):
    content: str
    author: str
    topic: Topic


def increment_count(count: str):
    return str(count) + "This is demo"


async def event_generator(request: Request, message: str) -> AsyncGenerator[dict, None]:
    try:
        llm = LLM()
        stream = llm.get_function_calling_response(query=message, response_format=Response, stream=True)

        field_processors = {
            "count": increment_count
        }
        processor = JSONStreamProcessor(field_processors)
        async for result in processor.process_stream(stream, request):
            yield result

    except Exception as e:
        yield {
            "event": "error",
            "data": str(e)
        }
        yield {
            "event": "end",
            "data": "failed"
        }


@router.post("/")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """Endpoint for streaming chat responses."""
    return EventSourceResponse(event_generator(request, chat_request.message))
