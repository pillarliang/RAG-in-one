import json

from fastapi import APIRouter
from pydantic.v1 import BaseModel
import redis

router = APIRouter(prefix="/chat", tags=["chat"])

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)


class UserQuery(BaseModel):
    user_id: str
    question: str


class ChatContext(BaseModel):
    question: str
    answer: str


def get_chat_context(user_id: str) -> list(ChatContext):
    context = redis_client.get(user_id)

    if context:
        return json.loads(context)
    else:
        # 如果 Redis 中没有数据，从 PostgreSQL 中获取最近的上下文
        pg_cursor.execute(
            """
            SELECT question, answer FROM chat_history
            WHERE user_id = %s
            ORDER BY id DESC LIMIT 5
            """,
            (user_id,)
        )
        rows = pg_cursor.fetchall()
        context = [{"question": row[0], "answer": row[1]} for row in rows]
        return context


def generate_answer(question: str) -> str:
    return f"This is the answer to: {question}"


def update_chat_context(user_id: str, question: str, answer: str) -> None:
    context = get_chat_context(user_id)
    context.append(ChatContext(question=question, answer=answer))
    redis_client.set(user_id, json.dumps(context), ex=3600)


def save_chat_to_db(user_id: str, question: str, answer: str) -> None:
    pg_cursor.execute(
        """
        INSERT INTO chat_history (user_id, question, answer)
        VALUES (%s, %s, %s)
        """,
        (user_id, question, answer)
    )
    pg_conn.commit()


async def chat(user_query: UserQuery) -> dict:
    user_id = user_query.user_id
    question = user_query.question

    # get context
    context = get_chat_context(user_id)

    # add new question to context
    context_text = " ".join([f"Q: {item['question']} A: {item['answer']}" for item in context])
    context_text += f" Q: {question}"

    # call LLM to generate answer
    answer = generate_answer(context_text)

    # update redis context
    update_chat_context(user_id, question, answer)

    # persist the chat to PostgreSQL
    save_chat_to_db(user_id, question, answer)

    return {"answer": answer}

# 关闭 PostgreSQL 连接
@app.on_event("shutdown")
def shutdown_event() -> None:
    pg_cursor.close()
    pg_conn.close()