from langchain_openai import ChatOpenAI
import os

# Config
API_KEY = "sk-624c20dbecac4a41913f1e66e83ea1ec"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-vl-max"

def test_init():
    print("Testing ChatOpenAI init with api_key and base_url...")
    try:
        llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0.1
        )
        print("Init success!")
        print(f"openai_api_key: {llm.openai_api_key}")
        print(f"openai_api_base: {llm.openai_api_base}")
    except Exception as e:
        print(f"Init failed: {e}")

if __name__ == "__main__":
    test_init()
