import json
from typing import Callable, Dict, List, Union
from types import SimpleNamespace
from model.llm import LLM


class Tool:
    def __init__(self, name: str, fn: Callable, fn_schema: str):
        self.name = name
        self.fn = fn
        self.fn_schema = fn_schema

    def __str__(self):
        return self.fn_schema

    def run(self, **kwargs):
        return self.fn(**kwargs)


class ToolAgent:
    # Class-level tool registry as a SimpleNamespace
    TOOLS = SimpleNamespace()

    @classmethod
    def register_tool(cls, tool: Tool):
        """Register a tool in the class-level registry as both a dictionary entry and an attribute."""
        # Register tool in the namespace (to allow dot access)
        setattr(cls.TOOLS, tool.name, tool)

    def __init__(self, tools: Union[Tool, List[Tool], None] = None):
        self.client = LLM()
        if tools is None:
            # Use all tools from the registry if none are provided
            self.tools = []
            for name in dir(ToolAgent.TOOLS):
                if not name.startswith("__"):
                    self.tools.append(getattr(ToolAgent.TOOLS, name))
        else:
            self.tools = tools if isinstance(tools, list) else [tools]

        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.tools_schema = [json.loads(tool.fn_schema) for tool in self.tools]

    def run(self, query: str) -> str:
        agent_chat_history = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the supplied tools to assist the user."
            },
            {"role": "user", "content": query}
        ]
        response = self.client.get_response_with_tools(
            tools=self.tools_schema, messages=agent_chat_history
        )
        fun_name = response.choices[0].message.tool_calls[0].function.name
        fun_args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

        tool = self.tools_dict[fun_name]
        tool_args = self._validate_arguments(fun_args, json.loads(tool.fn_schema))
        result = tool.run(**tool_args)
        agent_chat_history.append(
            {"role": "user", "content": f'The result obtained by calling the function is: {result}'}
        )
        res = self.client.get_response(agent_chat_history)
        return res

    def _validate_arguments(self, tool_call: dict, tool_schema: dict):
        """
        Validates if the arguments in the input dictionary match the types specified in the schema.
        """
        properties = tool_schema["function"]["parameters"]["properties"]
        type_mapping = {
            "int": int,
            "str": str,
            "bool": bool,
            "float": float,
        }

        for arg_name, arg_value in tool_call.items():
            expected_type = properties[arg_name].get("type")
            if not isinstance(arg_value, type_mapping[expected_type]):
                tool_call[arg_name] = type_mapping[expected_type](arg_value)

        return tool_call


def get_fn_schema(fn: Callable) -> Dict:
    """Generate a function signature dictionary from a callable."""
    fn_signature = {
        "name": fn.__name__,
        "description": fn.__doc__,
        "parameters": {
            "properties": {},
            "type": "object",
        }
    }
    schema = {k: {"type": v.__name__} for k, v in fn.__annotations__.items() if k != "return"}
    fn_signature["parameters"]["properties"] = schema

    fn_schema = {
        "type": "function",
        "function": fn_signature,
    }
    return fn_schema


def tool_use(fn: Callable) -> Callable:
    """Decorator to register a function as a Tool instance."""
    fn_schema = get_fn_schema(fn)
    tool_instance = Tool(
        name=fn_schema.get("function").get("name"),
        fn=fn,
        fn_schema=json.dumps(fn_schema)
    )
    ToolAgent.register_tool(tool_instance)
    return fn  # Return the original function unmodified


if __name__ == "__main__":
    # Example usage
    @tool_use
    def fake_func():
        """This is a fake function for demonstration purposes."""
        return "Fake function executed."


    # Access via TOOLS.<function_name> directly
    print(ToolAgent.TOOLS.fake_func)  # Accesses the tool directly as an attribute

    tool_agent = ToolAgent([ToolAgent.TOOLS.fake_func])
    output = tool_agent.run(query="notebooks")
    print(output)
