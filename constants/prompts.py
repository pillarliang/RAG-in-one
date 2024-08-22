"""
Author: pillar
Date: 2024-08-22
Description: prompts
Note: Different languages should use prompts that are appropriate for that language.
"""

EN_RAG_PROMPTS = """
CONTEXTS:
{contexts}

QUESTION:
{question}

INSTRUCTIONS:
Answer the users QUESTION using the CONTEXTS text above.
Keep your answer ground in the facts of the DOCUMENT.
If the [CONTEXTS] doesn’t contain the facts to answer the QUESTION return NONE.
"""

CN_RAG_PROMPTS = """
上下文:
{contexts}

问题:
{question}

说明:
根据上下文回答问题，如果上下文无法回答问题，请返回:[暂找不到相关问题，请重新提供问题。]
"""