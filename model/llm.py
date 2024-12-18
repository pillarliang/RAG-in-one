import os
import instructor
from typing import List, Optional, Type, Union
from instructor import OpenAISchema
from openai import OpenAI
from constants.type import LLMModel, MultiModalParameters
from utility.tools import batch_image_to_base64, PIL_2_base64, is_PIL_image, is_base64
from pydantic import BaseModel


class LLM:
    def __init__(self, api_key: Optional[str] = "sk-Fr7Bl02uYf5jXnkl4190783cFc414c68A2Fa75B68064FcDc", base_url: Optional[str] = "https://aihubmix.com/v1", model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._check_valid()

        self.model = model or LLMModel.Default.value
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.structured_client = instructor.from_openai(OpenAI(api_key=api_key, base_url=base_url), mode=instructor.Mode.JSON_SCHEMA)

    def get_response(self, query: Union[str, List[str]], response_format: Type[BaseModel] = None):
        if response_format:
            structured_res = self.structured_client.chat.completions.create(
                model=LLMModel.Default.value,
                messages=self._get_messages_for_llm(query),
                response_model=response_format,
            )
            return structured_res
        else:
            completion = self.client.chat.completions.create(
                model=LLMModel.Default.value,
                messages=self._get_messages_for_llm(query)
            )
            return completion.choices[0].message.content

    def get_function_calling_response(self, query: Union[str, List[str]], response_format: Type[BaseModel], stream=False):
        """
        根据 pydantic 模型, 生成 function calling 对应的 OpenAI Schema。
        主要用于获取 json 对象。
        """
        class FunctionSchema(response_format, OpenAISchema):
            pass

        func_schema = {
            "type": "function",
            "function": FunctionSchema.openai_schema,
        }

        return self.client.chat.completions.create(
            model=LLMModel.Default.value,
            messages=self._get_messages_for_llm(query),
            stream=stream,
            tools=[func_schema],
            tool_choice="required"
        )

    def get_response_with_tools(self, tools: list, query: Union[str, List[str]]):
        """主要用与根据实际的函数生成的 schema """
        try:
            response = self.client.chat.completions.create(
                model=LLMModel.Default.value,
                messages=self._get_messages_for_llm(query),
                tools=tools,
                tool_choice="required"
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e

    def get_json_mode_response(self, query: Union[str, List[str]]):
        return self.client.chat.completions.create(
            model=LLMModel.Default.value,
            messages=self._get_messages_for_llm(query),
            response_format={"type": "json_object"},
        )

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
                messages.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{PIL_2_base64(item)}"}})
        elif images and is_base64(images[0]):
            for item in images:
                messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{item}"}})
        elif images:
            for item in batch_image_to_base64(images):
                messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{item}"}})

        completion = self.client.chat.completions.create(
            model=LLMModel.Default.value,
            messages=[
                {
                    "role": "user",
                    "content": messages
                },
            ],
        )

        return completion.choices[0].message.content

    @staticmethod
    def _get_messages_for_llm(query: Union[str, List[str]]):
        if isinstance(query, list):
            messages = query
        else:
            messages = [{"role": "user", "content": query}]
        return messages

    def _check_valid(self):
        if not self.api_key:
            raise ValueError(
                "API key is required. Please provide it as a parameter or set the OPENAI_API_KEY environment variable.")
        if not self.base_url:
            raise ValueError(
                "Base URL is required. Please provide it as a parameter or set the OPENAI_BASE_URL environment variable.")

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
            model=LLMModel.Default.value,
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

    # res = llm.get_multimodal_response("请描述图片中的内容", {"images": ["pig.jpg"]})
    # print(res)
