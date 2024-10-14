# Agentic Design Patterns

我们要挖掘和利用 LLM 涌现出的能力。利用 LLM 能力(也可以概括为 Agent)的两种方式如下，：

1. In context leanring

2. Chain of thought

   

提高 GPT-4 和 GPT-3.5 性能的四种 AI agent策略

- Reflection: LLM 检查自己的工作，以寻找改进的方法。
- Tool Use: LLM 被赋予了诸如网络搜索、代码执行或其他任何功能的工具，以帮助其收集信息、采取行动或处理数据。
- Planning: LLM 制定并执行一个多步骤计划以实现目标（例如，撰写论文大纲，然后进行在线研究，然后撰写草稿，等等）。
- Multi-agent collaboration: 多个 AI Agent 共同工作，分配任务并讨论和辩论想法，以提出比单个Agent 更好的解决方案。



## 1. Reflection

这种模式允许大型语言模型（LLMs）对其输出进行反思和批评，遵循以下步骤：

1. LLMs 生成一个候选输出。
2. LLMs 对之前的输出进行反思，提出修改、删除、改进写作风格等建议。
3. LLMs 根据反思对原始输出进行修改，然后开始新一轮迭代……

> 在使用 ChatGPT 时，如果 reponse 不满意，我们会提供反馈来帮助 LLM 改进其输出，获得更好的 response。这里目的是：自动化提供反馈的过程。

情景：撰写行业简评。在生成初稿后，agent 对其进行阅读，找出需要修改的部分，然后通过反复优化进行完善。

[code:](../core/agentic_patterns/reflection.py)

[demo:](../notebooks/agentic_patterns/reflection.ipynb)



## 2. TOOL USE

LLM被赋予可以请求调用以收集信息、采取行动或操纵数据的功能。

具体过程：LLM要么经过微调，要么被提示（可能是通过少量提示）生成一个特殊字符串，如 `{tool: web-search, query: "咖啡机评论"}` （字符串的确切格式取决于实现。）以请求调用搜索引擎。当找到搜索引擎函数时时，使用相关参数调用该函数，并将结果作为额外输入上下文传递回LLM以进行进一步处理。

<img src="https://www.deeplearning.ai/_next/image/?url=https%3A%2F%2Fdl-staging-website.ghost.io%2Fcontent%2Fimages%2F2024%2F04%2Funnamed---2024-04-03T140654.796-2.png&w=3840&q=75" alt="Agentic Design Patterns Part 3, Tool Use: How large language models can act as agents by taking advantage of external tools for search, code execution, productivity, ad infinitum" style="zoom:40%;" />



情景：为 Instagram 等平台生成旅行指南。Agent 可以搜索有关当地天气、交通路线和景点开放时间的实时文本和视觉信息，同时根据作者的风格和平台标准编辑和格式化内容，从而实时制作高质量的内容。

[code:](../core/agentic_patterns/tool_use.py)

[demo:](../notebooks/agentic_patterns/tool_use.ipynb)


## 3. Planning Mode

 Agent 动态将复杂任务分解并按照计划执行。另一方面，它导致结果的可预测性降低。

- 背景：大型模型的生成效果取决于训练数据的有效性，有时可能由于幻觉而产生次优结果。

- 规划模式允许 agent 基于计划的任务步骤对生成的内容进行反复优化和处理，从而产生更高质量的输出。





## 4. Multiagent Collaboration Mode

多个AI Agent 协同工作，分配任务、讨论和辩论想法，以提出比单个 Agent 更优的解决方案。

- 背景：大型模型有时会遇到需要团队合作才能完成的系统性任务，而单个 Agent 通常只关注特定的能力。
- 情景：用户只需用自然语言填写配置文件，就可以轻松为各种功能和用例定义多代理系统，特别是在涉及不同角色（如编剧/小说创作）的内容创作工作室中。









Reference:

https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/