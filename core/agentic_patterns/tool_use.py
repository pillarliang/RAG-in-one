import json
from typing import Callable, Dict, List, Union
from model.llm import LLM


def get_fn_signature(fn: Callable) -> Dict:
    """Generate a function signature dictionary from a callable."""
    fn_signature = {
        "name": fn.__name__,
        "description": fn.__doc__,
        "parameters": {
            "properties": {}
        }
    }
    schema = {k: {"type": v.__name__} for k, v in fn.__annotations__.items() if k != "return"}
    fn_signature["parameters"]["properties"] = schema
    return fn_signature


class Tool:
    def __init__(self, name: str, fn: Callable, fn_signature: str):
        self.name = name
        self.fn = fn
        self.fn_signature = fn_signature

    def __str__(self):
        return self.fn_signature

    def run(self, **kwargs):
        return self.fn(**kwargs)


def tool(fn: Callable) -> Tool:
    """Decorator to create a Tool instance from a function."""
    def wrapper():
        fn_signature = get_fn_signature(fn)
        return Tool(
            name=fn_signature.get("name"),
            fn=fn,
            fn_signature=json.dumps(fn_signature)
        )
    return wrapper()


class ToolAgent:
    def __init__(self, tools: Union[Tool, List[Tool]]):
        self.client = LLM()
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}

    def add_tool_signatures(self) -> str:
        """Concatenate tool signatures into a single string."""
        return "".join(tool.fn_signature for tool in self.tools)

    def run(self, query: str) -> str:
        agent_chat_history = [{"role": "system",
                     "content": "You are a helpful assistant. Use the supplied tools to assist the user."},
                    {"role": "user", "content": query}]
        response = self.client.get_response_with_tools(tools=self.tools, messages=agent_chat_history)
        fun_name = response.choices[0].message.tool_calls[0].function["name"]
        fun_args = json.loads(response.choices[0].message.tool_calls[0].function["arguments"])

        tool = self.tools_dict[fun_name]
        tool_args = self.validate_arguments(fun_args, json.loads(tool.fn_signature))
        result = tool.run(**tool_args["arguments"])
        agent_chat_history.append({"role": "user", "content": f'The result obtained by calling the function is: {result}'})
        res = self.client.get_response(agent_chat_history)
        return res

    def validate_arguments(self, tool_call: dict, tool_signature: dict):
        """
        Validates if the arguments in the input dictionary match the types specified in the schema.
        """
        properties = tool_signature["parameters"]["properties"]

        type_mapping = {
            "int": int,
            "str": str,
            "bool": bool,
            "float": float,
        }

        for arg_name, arg_value in tool_call["arguments"].items():
            expected_type = properties[arg_name].get("type")

            if not isinstance(arg_value, type_mapping[expected_type]):
                tool_call["arguments"][arg_name] = type_mapping[expected_type](arg_value)

        return tool_call