from openai import OpenAI
import os
from constants.type import LLMModel


class DMetaLLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)  # temporarily using openai service

    def get_response(self, query: str):
        print("base_url: ", self.base_url)
        completion = self.client.chat.completions.create(
            model=LLMModel.Default.value,
            messages=[
                {"role": "user", "content": query},
            ],
        )
        return completion.choices[0].message.content

    def get_structured_response(self, query: str, response_format):
        completion = self.client.beta.chat.completions.parse(
            model=LLMModel.Default.value,  # TODO: currently using a specific model, due to only lasted model support structured response
            messages=[
                {"role": "user", "content": query}
            ],
            response_format=response_format,
        )

        return completion.choices[0].message.content


if __name__ == "__main__":
    llm = DMetaLLM()
    res = llm.get_response("What is the capital of China?")
    print(res)
