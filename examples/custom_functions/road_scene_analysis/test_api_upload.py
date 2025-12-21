import requests
import argparse
import os

def test_upload(image_path, url="http://localhost:8000/api/v1/upload-image"):
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return

    print(f"Uploading {image_path} to {url}...")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'location': '116.4074,39.9042',
                'device_id': 'test_client_01',
                'analysis_type': 'all'
            }
            
            response = requests.post(url, files=files, data=data)
            
        if response.status_code == 200:
            print("Upload successful!")
            print("Response:", response.json())
        else:
            print(f"Upload failed with status code {response.status_code}")
            print("Response:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure api_server.py is running.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test image upload to Traffic API Server")
    parser.add_argument("image_path", help="Path to the image file to upload")
    parser.add_argument("--url", default="http://localhost:8000/api/v1/upload-image", help="API endpoint URL")
    
    args = parser.parse_args()
    
    test_upload(args.image_path, args.url)
