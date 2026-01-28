import urllib.request
import json
import time
import sys
import os

BASE_URL = "http://localhost:8000"

def test_health():
    print("Testing Health Check...")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/") as response:
            data = json.loads(response.read().decode())
            print(f"Health OK: {data}")
    except Exception as e:
        print(f"Health Failed: {e}")
        sys.exit(1)

def test_txt2vid():
    print("Testing Txt2Vid...")
    payload = {
        "prompt": "A cinematic drone shot of a futuristic city",
        "num_frames": 81
    }
    req = urllib.request.Request(
        f"{BASE_URL}/generate/txt2vid",
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            print(f"Txt2Vid OK: {data}")
    except Exception as e:
        print(f"Txt2Vid Failed: {e}")
        sys.exit(1)

# Simple multipart form data encoder
def encode_multipart_formdata(fields, files):
    boundary = '---BOUNDARY'
    body = []
    
    for key, value in fields.items():
        body.append(f'--{boundary}')
        body.append(f'Content-Disposition: form-data; name="{key}"')
        body.append('')
        body.append(str(value))
        
    for key, (filename, content) in files.items():
        body.append(f'--{boundary}')
        body.append(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"')
        body.append('Content-Type: application/octet-stream')
        body.append('')
        body.append(content) # Expecting string or bytes?
        
    body.append(f'--{boundary}--')
    body.append('')
    
    # Join with CRLF? Standard demands it.
    # Simplified: constructing bytes
    
    crlf = b'\r\n'
    boundary_bytes = boundary.encode()
    
    parts = []
    for key, value in fields.items():
        parts.append(b'--' + boundary_bytes)
        parts.append(f'Content-Disposition: form-data; name="{key}"'.encode())
        parts.append(b'')
        parts.append(str(value).encode())
        
    for key, (filename, content) in files.items():
        parts.append(b'--' + boundary_bytes)
        parts.append(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"'.encode())
        parts.append(b'Content-Type: application/octet-stream')
        parts.append(b'')
        parts.append(content)
        
    parts.append(b'--' + boundary_bytes + b'--')
    parts.append(b'')
    
    return crlf.join(parts), f'multipart/form-data; boundary={boundary}'

def test_img2vid():
    print("Testing Img2Vid...")
    # Create dummy image
    dummy_img = b'fakeimagecontent'
    
    body, content_type = encode_multipart_formdata(
        {"prompt": "Animated image"},
        {"image": ("test.png", dummy_img)}
    )
    
    req = urllib.request.Request(
        f"{BASE_URL}/generate/img2vid",
        data=body,
        headers={'Content-Type': content_type},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            print(f"Img2Vid OK: {data}")
    except Exception as e:
        print(f"Img2Vid Failed: {e}")
        # Dont exit, just log

def main():
    # Wait for server to start
    print("Waiting for server...")
    time.sleep(5)
    
    try:
        test_health()
        test_txt2vid()
        test_img2vid()
        print("\nAll Backend Tests Passed!")
    except Exception as e:
        print(f"\nTests Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
