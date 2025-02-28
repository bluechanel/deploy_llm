# Multi-Model Deployment Version

The example file simultaneously deploys three LLMs: **DeepSeek-R1-Distill-Qwen-32B**, **Qwen2.5-72B-Instruct-GPTQ-Int4**, and **Qwen2.5-VL-7B-Instruct**, with traffic routed through an **OpenResty** proxy.

The invocation method is as follows:
```python
import os
from openai import OpenAI
import cohere

base_url = "http://127.0.0.1:8000/v1"

def get_openai_client():
    return OpenAI(
        api_key=os.getenv("API_KEY", "sk-example"),
        base_url=base_url
    )

def test_chat_completion():
    try:
        client = get_openai_client()
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say this is a test"}],
            model="deepseek-r1-qwen",
        )
        print(chat_completion.model_dump_json())
    except Exception as e:
        print(f"Chat completion error: {str(e)}")

def test_chat_completion1():
    try:
        client = get_openai_client()
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say this is a test"}],
            model="qwen2.5",
        )
        print(chat_completion.model_dump_json())
    except Exception as e:
        print(f"Chat completion error: {str(e)}")

def test_chat_completion2():
    try:
        client = get_openai_client()
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say this is a test"}],
            model="qwen2.5vl",
        )
        print(chat_completion.model_dump_json())
    except Exception as e:
        print(f"Chat completion error: {str(e)}")

def test_embedding():
    try:
        client = get_openai_client()
        embedding = client.embeddings.create(
            input="Say this is a test",
            model="gte-large-zh"
        )
        print(embedding.model_dump_json())
    except Exception as e:
        print(f"Embedding error: {str(e)}")

def test_rerank():
    try:
        co_client = cohere.ClientV2(
            api_key=os.getenv("API_KEY", "sk-example"),
            base_url=base_url.rstrip('/v1')
        )
        reranker = co_client.rerank(
            model='bge-reranker-base',
            query='Say this is a test',
            documents=["Say this is a test", "Say this is a test message"]
        )
        print(reranker.model_dump_json())
    except Exception as e:
        print(f"Rerank error: {str(e)}")

if __name__ == '__main__':
    test_chat_completion()
    test_chat_completion1()
    test_chat_completion2()
    test_embedding()
    test_rerank()
```

For more information, please visit: https://www.wileyzhang.com/llm%E9%83%A8%E7%BD%B2dockervllmembeddingrerank-%E6%94%AF%E6%8C%81%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8

