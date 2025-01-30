import requests
import json
from typing import Optional, Tuple, List, Dict
import sys
from tabulate import tabulate

class OllamaConnection:
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/api"
        self.results: List[Dict] = []
        self.verify_connection()
    
    def verify_connection(self):
        """Verify connection to Ollama server and print version info"""
        try:
            # Get IP address
            try:
                import socket
                ip_address = socket.gethostbyname(self.host)
            except socket.gaierror:
                ip_address = "Unable to resolve IP"

            response = requests.get(f"{self.base_url}/version", timeout=5)  # Add timeout
            if response.status_code == 200:
                version_info = response.json()
                print("\nOllama Server Connection Info:")
                print("─" * 40)
                print(f"Status:         Connected")
                print(f"Host:           {self.host}")
                print(f"IP Address:     {ip_address}")
                print(f"Port:           {self.port}")
                print(f"Ollama Version: {version_info.get('version', 'unknown')}")
                print(f"API URL:        {self.base_url}")
                print("─" * 40)
            else:
                raise ConnectionError(f"Server returned status code: {response.status_code}")
        except requests.Timeout:
            print("\nError: Connection timed out!")
            print(f"Could not connect to Ollama server at {self.host}:{self.port}")
            print("\nPossible solutions:")
            print("1. Make sure Ollama is running")
            print("2. Check if the host and port are correct")
            print("3. If using a remote server, ensure the server is accessible")
            print("\nTo start Ollama locally, run: ollama serve")
            sys.exit(1)
        except requests.ConnectionError:
            print("\nError: Could not connect to Ollama server!")
            print(f"Failed to establish connection to {self.host}:{self.port}")
            print("\nPossible solutions:")
            print("1. Make sure Ollama is running")
            print("2. Check if the host and port are correct")
            print("3. If using a remote server, ensure the server is accessible")
            print("\nTo start Ollama locally, run: ollama serve")
            sys.exit(1)
        except Exception as e:
            print(f"\nUnexpected error while connecting to Ollama server:")
            print(f"Error details: {str(e)}")
            print("\nPlease check your configuration and try again.")
            sys.exit(1)

    def format_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format (MB/GB)"""
        mb = size_bytes / (1024 * 1024)
        if mb >= 1024:
            return f"{mb/1024:.2f} GB"
        return f"{mb:.2f} MB"

    def list_models(self) -> list:
        """Get list of available models"""
        response = requests.get(f"{self.base_url}/tags")
        
        if response.status_code == 200:
            data = response.json()
            models = data["models"]
            print(f"\nFound {len(models)} model(s) on the server")
            return models
        else:
            print(f"Error: Failed to retrieve models. Status code: {response.status_code}")
            return []

    def measure_model_speed(self, model_name: str) -> Tuple[float, dict]:
        """Measure speed for a specific model"""
        url = f"{self.base_url}/generate"
        payload = {
            "model": model_name,
            "prompt": "Why is the sky blue?",
            "stream": False
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response_data = response.json()
        
        eval_count = response_data.get("eval_count", 0)
        eval_duration = response_data.get("eval_duration", 1)  # Prevent division by zero
        eval_duration_seconds = eval_duration / 1e9
        speed = eval_count / eval_duration_seconds
        
        return speed, response_data

    def add_result(self, model_name: str, size: str, family: str, parameters: str, speed: float, tokens: int):
        """Store a test result"""
        self.results.append({
            "Model": model_name,
            "Size": size,
            "Family": family,
            "Parameters": parameters,
            "Speed (t/s)": f"{speed:.2f}",
            "Tokens": tokens
        })

    def print_results_table(self):
        """Print results in a table format"""
        if not self.results:
            return
        
        print("\nTest Results Summary:")
        print("─" * 100)
        print(tabulate(self.results, headers="keys", tablefmt="grid"))
        print("─" * 100)

def main(host: Optional[str] = None, port: Optional[int] = None):
    # Use default values if not provided
    host = host or "localhost"
    port = port or 11434
    
    try:
        # Initialize connection
        ollama = OllamaConnection(host, port)
        
        # Get models
        models = ollama.list_models()
        total_models = len(models)
        
        # Measure speed for each model
        for index, model in enumerate(models, 1):
            model_name = model["name"]
            print(f"\nTesting model ({index}/{total_models}): {model_name}")
            print("Model details:")
            size = ollama.format_size(model['size'])
            details = model['details']
            family = details.get('family', 'unknown')
            parameters = details.get('parameter_size', 'unknown')
            
            print(f"  Size:           {size}")
            print(f"  Family:         {family}")
            print(f"  Parameters:     {parameters}")
            
            try:
                speed, response_data = ollama.measure_model_speed(model_name)
                tokens = response_data.get('eval_count', 0)
                
                print(f"Performance metrics:")
                print(f"  Speed:          {speed:.2f} tokens/second")
                print(f"  Total tokens:   {tokens}")
                print(f"  Response length: {len(response_data.get('response', ''))}")
                
                # Store the result
                ollama.add_result(model_name, size, family, parameters, speed, tokens)
                
            except Exception as e:
                print(f"  Error measuring speed: {str(e)}")
        
        # Print results table at the end
        ollama.print_results_table()
                
    except ConnectionError as e:
        print(f"Connection error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Measure Ollama model speeds')
    parser.add_argument('--host', type=str, help='Ollama server host (default: localhost)')
    parser.add_argument('--port', type=int, help='Ollama server port (default: 11434)')
    
    args = parser.parse_args()
    main(args.host, args.port)