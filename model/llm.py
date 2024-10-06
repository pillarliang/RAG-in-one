import json
import os
from typing import List, Optional
from openai import OpenAI
from constants.type import LLMModel, MultiModalParameters
from utility.tools import batch_image_to_base64, PIL_2_base64, is_PIL_image, is_base64_image, is_base64

os.environ["OPENAI_API_KEY"] = "sk-sduyrRdYYOdGP4x06e97DdDe7bA74c7e8a5aC1051d5a2831"
os.environ["OPENAI_BASE_URL"] = "https://aihubmix.com/v1"


class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)  # temporarily using openai service

    def get_response(self, query: Optional[str, List[str]]):
        completion = self.client.chat.completions.create(
            model=LLMModel.Default.value,
            messages=self._get_messages_for_llm(query),
        )
        return completion.choices[0].message.content

    def get_structured_response(self, query: str, response_format):
        completion = self.client.beta.chat.completions.parse(
            model=LLMModel.Default.value,
            messages=self._get_messages_for_llm(query),
            response_format=response_format,
        )

        return json.loads(completion.choices[0].message.content)

    def get_multimodal_response(self, query: str, contexts: MultiModalParameters):
        texts = contexts.get("texts", "")
        images = contexts.get("images", "")
        prompts = f"""
        请根据文本【texts】以及图片回答问题【question】。
        - 如果根据所提供的【texts】和图片信息无法回答问题【question】，请返回:暂找不到相关问题，请重新提供问题。
        - 如果根据所提供的【texts】和图片信息能够回答问题【question】，请你给出结果。
        
        question: {query}
        
        texts: {texts}
        """

        messages = [{"type": "text", "text": prompts}]

        if images and is_PIL_image(images[0]):
            for item in images:
                messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{PIL_2_base64(item)}"}})
        elif images and is_base64(images[0]):
            for item in images:
                messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{item}"}})
        elif images:
            for item in batch_image_to_base64(images):
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

    def _get_messages_for_llm(self, query: Optional[str, List[str]]):
        if isinstance(query, list):
            messages = query
        else:
            messages = [{"role": "user", "content": query}]
        return messages

    def get_response_with_tools(self, tools: list, messages: list):
        response = self.client.chat.completions.create(
            model=LLMModel.Default.value,
            messages=messages,
            tools=tools,
            tool_choice="required"
        )
        return response

    def ask_images(self, query: str, images: List[str]):
        """Temporarily used for testing"""
        prompts = f"""
                请根据图片回答问题【question】。
                - 如果根据所提供的【texts】和图片信息无法回答问题【question】，请返回:暂找不到相关问题，请重新提供问题。
                - 如果根据所提供的【texts】和图片信息能够回答问题【question】，请你给出结果。

                question：{query}

                """

        messages = [{"type": "text", "text": prompts}]
        for item in images:
            messages.append({"type": "image_url", "image_url": {"url": item}})

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
    llm = LLM()
    res = llm.get_response("What is the capital of China?")
    print(res)

    res = llm.get_multimodal_response("请描述图片中的内容", {"images": ["pig.jpg"]})
    print(res)
