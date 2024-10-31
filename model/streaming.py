from typing import Generator, Any, Dict, Callable, Optional
import streamingjson


class JSONStreamProcessor:
    def __init__(self, field_processors: Dict[str, Callable[[str], str]] = None):
        """
        初始化处理器
        Args:
            field_processors: 字段处理器字典，键为字段名，值为处理该字段值的函数
        """
        self.lexer = streamingjson.Lexer()
        self.current_key = None
        self.previous_chunk = ""
        self.collecting_field_value = False
        self.value_buffer = ""
        self.field_processors = field_processors or {}
        self.current_processor = None
        self.complete_response = ""

    @classmethod
    def is_json_field_end(cls, chunk: str) -> bool:
        return chunk.startswith('",') or chunk.endswith("}")

    @classmethod
    def is_key_indicator(cls, chunk: str) -> bool:
        """检查当前chunk是否为\":\"，这表示前一个chunk是一个键"""
        return chunk.startswith('":')

    async def process_stream(self, stream: Generator, request: Any) -> Generator[dict, None, None]:
        is_llm_response = False

        for chunk in stream:
            # 检查客户端连接状态
            if await request.is_disconnected():
                break

            # 处理工具调用为None的情况
            if chunk.choices[0].delta.tool_calls is None and is_llm_response:
                self.complete_response = self.lexer.complete_json()
                yield {
                    "event": "end",
                    "data": "success"
                }
                break

            if chunk.choices[0].delta.tool_calls[0].function.arguments is not None:
                is_llm_response = True
                response_chunk = chunk.choices[0].delta.tool_calls[0].function.arguments
                print(response_chunk)

                # 如果没有需要处理的字段，直接追加内容并继续
                if not self.field_processors:
                    self.lexer.append_string(response_chunk)
                    complete_json = self.lexer.complete_json()
                    yield {
                        "event": "message",
                        "data": response_chunk
                    }
                    continue

                # 如果正在收集字段的值
                if self.collecting_field_value:
                    if self.is_json_field_end(response_chunk):
                        # 值收集完成，进行处理
                        processed_value = self.current_processor(self.value_buffer)
                        # 替换lexer中的值
                        self.lexer.append_string(f"{processed_value}{response_chunk}")
                        self.collecting_field_value = False
                        self.current_processor = None

                        # 收集完成后，生成完整的JSON响应并yield
                        # complete_json = self.lexer.complete_json()
                        yield {
                            "event": "message",
                            "data": f"{processed_value}{response_chunk}"
                        }
                    else:
                        # 继续收集值
                        if not self.is_key_indicator(
                                response_chunk) and response_chunk != "null":  # null 是streamingjson 中的，与 Python 的 None 不同
                            self.value_buffer += response_chunk
                        # 在收集过程中，不yield
                else:
                    # 非收集状态，直接将chunk加入lexer
                    self.lexer.append_string(response_chunk)
                    # 检测是否是键标识符并且有字段需要处理
                    if self.is_key_indicator(response_chunk):
                        # 前一个chunk就是键
                        self.current_key = self.previous_chunk
                        if self.current_key in self.field_processors:
                            self.collecting_field_value = True
                            self.value_buffer = ""
                            self.current_processor = self.field_processors[self.current_key]

                    # 非收集状态，生成完整的JSON响应并yield
                    # complete_json = self.lexer.complete_json()
                    yield {
                        "event": "message",
                        "data": response_chunk
                    }

                # 更新previous_chunk用于下一次检测
                self.previous_chunk = response_chunk
