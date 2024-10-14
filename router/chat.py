import json
import logging
import os
from contextlib import asynccontextmanager

import redis
from fastapi import APIRouter, FastAPI
from py_nl2sql.constants.type import RDBType
from py_nl2sql.relational_database import create_rdb
from pydantic import BaseModel
from sqlalchemy import String, Text
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import Text, TIMESTAMP, String, func
from model.llm import LLM


router = APIRouter(prefix="/chat", tags=["chat"])
# ##### Database start #####
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
postgres_db_client = create_rdb(
    db_type=RDBType.Postgresql,
    db_name=os.getenv("DB_NAME", "chat"),
    db_host=os.getenv("DB_HOST", "localhost"),
    db_user=os.getenv("DB_USER", "liangzhu"),
    db_password=os.getenv("DB_PASSWORD", ""),
)

Base = declarative_base()


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(TIMESTAMP, server_default=func.now())


Base.metadata.create_all(postgres_db_client.engine)
# ##### Database end #####

logger = logging.getLogger(__name__)


class UserQuery(BaseModel):
    user_id: str
    question: str


@router.get("/")
async def test():
    return {"message": "Hello, World!"}


@router.post("/")
async def chat(user_query: UserQuery) -> dict:
    user_id = user_query.user_id
    question = user_query.question

    context = get_chat_context(user_id)
    context_prompt = " ".join([f"Q: {item['question']} A: {item['answer']}" for item in context]) if context else ""
    context_prompt += f" Q: {question}"

    # call LLM to generate answer
    llm = LLM()
    answer = llm.get_response(context_prompt)
    logger.info(f"answer from llm:{answer}")

    # update redis context
    update_chat_context(user_id, question, answer)

    # update PostgreSQL chat history
    save_chat_to_db(user_id, question, answer)

    return {"answer": answer}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    postgres_db_client.engine.dispose()


def get_chat_context(user_id: str, limit: int = 3):
    """chat context redis cache"""
    context = redis_client.get(user_id)
    logger.info(f"context in redis:{context}")
    if context:
        context = json.loads(context)
        if len(context) < limit:
            # 如果 Redis 中的数据不足，从 PostgreSQL 中补充剩余的上下文
            with postgres_db_client.Session() as db:
                rows = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).order_by(ChatHistory.id.desc()).limit(limit - len(context)).all()
                additional_context = [{"question": row.question, "answer": row.answer} for row in rows]
                context.extend(additional_context)
                logger.info(f"context in postgres:{context}")
    else:
        # 如果 Redis 中没有数据，从 PostgreSQL 中获取最近的上下文
        with postgres_db_client.Session() as db:
            rows = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).order_by(ChatHistory.id.desc()).limit(
                limit).all()
            context = [{"question": row.question, "answer": row.answer} for row in rows]
            logger.info(f"context in postgres:{context}")
    return context


def update_chat_context(user_id: str, question: str, answer: str):
    context = get_chat_context(user_id)
    context.append({"question": question, "answer": answer})
    redis_client.set(user_id, json.dumps(context), ex=3600)


def save_chat_to_db(user_id: str, question: str, answer: str):
    with postgres_db_client.Session() as db:
        chat_record = ChatHistory(user_id=user_id, question=question, answer=answer)
        logger.info(f"chat_record written in postgresql:{chat_record}")
        db.add(chat_record)
        db.commit()
