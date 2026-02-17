import urllib.request
import json
import time
import urllib.parse

API_URL = "http://localhost:8001"

def post_request(endpoint, data):
    url = f"{API_URL}{endpoint}"
    # Convert data to form-urlencoded format as expected by Form(...) parameters in FastAPI
    # However, for non-nested data, this works. For nested JSON strings, we need to ensure correct encoding.
    # The API expects Form data.
    encoded_data = urllib.parse.urlencode(data).encode('utf-8')
    req = urllib.request.Request(url, data=encoded_data, method='POST')
    try:
        with urllib.request.urlopen(req) as f:
            resp = f.read().decode('utf-8')
            return f.status, json.loads(resp)
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
        return e.code, None
    except Exception as e:
        print(f"Request Exception: {e}")
        return 0, None

def reproduce_crash():
    print("Starting reproduction (urllib)...")
    
    # 1. Start Session with dummy video (121 frames)
    video_path = "<load-dummy-video-500>"
    print(f"Starting session for {video_path}...")
    
    status, res = post_request("/track/start", {"video_path": video_path})
    print(f"Start Status: {status}")
    if status != 200 or not res:
        print("Start Failed")
        return
    session_id = res["session_id"]
    print(f"Session ID: {session_id}")

    # 2. Add Prompt to Frame 0
    print("Adding prompt to frame 0...")
    status, res = post_request("/track/prompt", {
        "session_id": session_id,
        "frame_idx": 0,
        "points": json.dumps([[50, 50]]),
        "point_labels": json.dumps([1]),
        "obj_id": 1
    })
    print(f"Prompt Status: {status}")
    if status != 200:
        print("Prompt Failed")
        return

    # 3. Propagate
    print("Propagating...")
    status, res = post_request("/track/propagate", {
        "session_id": session_id,
        "start_frame": 400,
        "max_frames": -1 # All frames
    })
    print(f"Propagate Status: {status}")
    if status == 200:
        print("Propagate Success (Crash NOT reproduced?)")
        if res:
             print(f"Frames processed: {res.get('frame_count')}")
    else:
        print("Propagate Failed (Potential Crash)")

if __name__ == "__main__":
    reproduce_crash()
