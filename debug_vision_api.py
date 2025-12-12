import base64
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Config
API_KEY = "sk-624c20dbecac4a41913f1e66e83ea1ec"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-vl-max"
IMAGE_PATH = "/home/orin/Documents/mec-state-video/alert_images/cam_zs_20250623_125748.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_analysis():
    print(f"Testing model: {MODEL_NAME}")
    print(f"Image path: {IMAGE_PATH}")
    
    try:
        # Encode image
        base64_image = encode_image(IMAGE_PATH)
        print(f"Image encoded. Length: {len(base64_image)}")
        
        llm = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            temperature=0.1,
            max_tokens=2048,
        )
        
        prompt = "请全面分析这张路侧场景图片。包括：交通状况（道路通畅度、车辆流量、交通标志、交通灯状态、事故等）、环境信息（建筑物、街道设施、地标、人流等）和天气条件（能见度、天气状况等）。"
        
        messages = [
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            )
        ]
        
        print("Sending request to LLM...")
        response = llm.invoke(messages)
        print("\n--- Success! ---")
        print(response.content)
        
    except Exception as e:
        print(f"\n--- Error ---")
        print(f"Type: {type(e)}")
        print(f"Message: {e}")

if __name__ == "__main__":
    test_analysis()
