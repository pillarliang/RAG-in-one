### Terms:

`chunks`: The text before retrieval, used in the retrieval phase.

`contexts`: The text after retrieval, used in the generation phase.

### run
```shell
docker build -t rag_demo .
docker run -p 8000:8000 --name rag_demo rag_demo
```
## Usage
### 1. Multimodal RAG
输入：
- query: str,  # 用户查询
- text: List[str], # 文本
- images: List[str], # 图像的文件路径

输出：answer

示例 demo：`test/multimodel_demo.ipynb`



## API
### 1. 根据 query 以及 chunks 获取答案

**URL:** `http://localhost:8000/query`

**方法:** `POST`

**描述:**
此接口用于根据用户的查询以及提供的 documents 原始 chunks来获取对应的答案，如果所提供的上下文没有答案，返回"暂找不到相关问题，请重新提供问题。"

**请求参数:**

- `query` (字符串): 用户查询的问题。
- `chunks` (字符串): 已切分好的文本 chunks。

**请求示例:**
```json
{
  "query":"小明的爱好是啥？",
  "chunks":["小明喜欢打蓝球", "小明和小红是同事", "小红的工作是一名演员"]
}
```
**响应示例:**
一个字符串
```json
"小明喜欢打蓝球"
```

### [Agentic Patterns](./docs/agentic-patterns.md)
四种 AI agent策略模式
