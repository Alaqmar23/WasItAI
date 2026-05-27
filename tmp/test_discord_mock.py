import requests
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import time

class MockDiscordHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        print("\n--- Received Discord Webhook Request ---")
        print(f"Content-Type: {self.headers['Content-Type']}")
        # We won't print the whole binary body, but we'll check if parts exist
        if b'name="payload_json"' in body:
            print("✅ found payload_json part")
        if b'name="file"' in body:
            print("✅ found file part")
        
        self.send_response(204)
        self.end_headers()

def run_mock_server():
    server = HTTPServer(('localhost', 9999), MockDiscordHandler)
    server.handle_request()

# Start mock server in thread
threading.Thread(target=run_mock_server).start()
time.sleep(1)

# The logic we want to test
def test_send_to_discord(url, prediction, image_id, file_contents, filename):
    label = "AI" if prediction == "AUTHENTIC" else "AUTHENTIC"
    payload = {
        "content": f"🚨 **Correction Required**\n**Scan:** {prediction}\n**Actual:** {label}\n**ID:** {image_id}"
    }
    
    # Correct way to send both JSON payload and file to Discord
    files = {
        "file": (filename, file_contents, "image/jpeg"),
        "payload_json": (None, json.dumps(payload))
    }
    
    try:
        r = requests.post(url, files=files, timeout=5)
        print(f"Status Code: {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

# Run the test
test_send_to_discord(
    "http://localhost:9999", 
    "AUTHENTIC", 
    "test_id_123", 
    b"fake_image_data", 
    "test.jpg"
)
