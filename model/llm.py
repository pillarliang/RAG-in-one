import json

from openai import OpenAI
import os
from constants.type import LLMModel
from utility.tools import batch_image_to_base64


class DMetaLLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)  # temporarily using openai service

    def get_response(self, query: str):
        completion = self.client.chat.completions.create(
            model=LLMModel.Default.value,
            messages=[
                {"role": "user", "content": query},
            ],
        )
        return completion.choices[0].message.content

    def get_structured_response(self, query: str, response_format):
        completion = self.client.beta.chat.completions.parse(
            model=LLMModel.Default.value,
            # TODO: currently using a specific model, due to only lasted model support structured response
            messages=[
                {"role": "user", "content": query}
            ],
            response_format=response_format,
        )

        return json.loads(completion.choices[0].message.content)

    def get_multimodal_response(self, query: str, contexts):
        texts = contexts.get("texts", "")
        images = contexts.get("images", "")
        encoded_images = batch_image_to_base64(images)

        prompts = f"""
        请根据所提供的上下文以及图片信息回答问题而不是先验知识来回答以下的查询。如果上下文无法回答问题，请返回:暂找不到相关问题，请重新提供问题。
        
        问题：{query}
        
        上下文：{texts}
        """

        messages = [{"type": "text", "text": prompts}]
        for item in encoded_images:
            messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{item}"}})

        completion = self.client.chat.completions.create(
            model=LLMModel.GPT_4o_mini.value,
            messages=[
                {
                    "role": "user",
                    "content": messages
                },
            ],
        )
        return completion.choices[0].message.content


if __name__ == "__main__":
    llm = DMetaLLM()
    res = llm.get_response("What is the capital of China?")
    print(res)
