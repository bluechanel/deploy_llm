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
            model="gpt-4",
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
    test_embedding()
    test_rerank()
