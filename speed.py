import requests
import json

def list_models():
    url = "http://localhost:11434/api/tags"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        models = data["models"]
        
        print("Models running on ollama server:")
        for model in models:
            name = model["name"]
            size = model["size"]
            details = model["details"]
            
            print(f"\nModel: {name}")
            print(f"Size: {size} bytes")
            print(f"Details:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        return [model["name"] for model in models]
    else:
        print(f"Error: Failed to retrieve models. Status code: {response.status_code}")
        return []

def get_version_info():
    url = "http://localhost:11434/api/version"
    response = requests.get(url)

    if response.status_code == 200:
        version_info = response.json()
        print(f"Ollama Version Information:")
        for key, value in version_info.items():
            print(f"{key}: {value}")
    else:
        print(f"Error: Failed to retrieve version information. Status code: {response.status_code}")


def send_request(model):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": "Why is the sky blue?",
        "stream": False  # Adjust this based on whether you want streaming or not
    }
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    return response.json()

def calculate_speed(response_data):
    eval_count = response_data.get("eval_count", 0)
    eval_duration = response_data.get("eval_duration", 1)  # Prevent division by zero

    # Convert nanoseconds to seconds for eval_duration
    eval_duration_seconds = eval_duration / 1e9

    # Calculate tokens per second
    speed = eval_count / eval_duration_seconds
    return speed

def main():
    get_version_info()
    
    models = list_models()

    for model in models:
        print(f"\nMeasuring performance for model: {model}")
        response_data = send_request(model)
        speed = calculate_speed(response_data)
        print(f"Speed: {speed} tokens/second")

if __name__ == "__main__":
    main()