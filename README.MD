
**Terms:**

`chunks`: The text before retrieval, used in the retrieval phase.

`contexts`: The text after retrieval, used in the generation phase.


## run
```shell
docker build -t rag_demo .
docker run -p 8000:8000 --name rag_demo rag_demo
```
