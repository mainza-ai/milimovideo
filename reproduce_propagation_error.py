
import requests
import time
import os

SAM_URL = "http://localhost:8001"
# VIDEO_PATH = "/Users/mck/Desktop/milimovideo/backend/projects/8f3f5037/generated/job_2d3a80c2.mp4"
# Assuming the file exists and backend/projects/8f3f5037/generated/job_2d3a80c2.mp4 is correct relative to CWD /Users/mck
VIDEO_PATH = "/Users/mck/Desktop/milimovideo/backend/projects/8f3f5037/generated/job_2d3a80c2.mp4"

def test_propagation():
    print(f"Starting session on {VIDEO_PATH}")
    res = requests.post(f"{SAM_URL}/track/start", data={"video_path": VIDEO_PATH})
    print(f"Start status: {res.status_code}")
    if res.status_code != 200:
        print(res.text)
        return
    
    data = res.json()
    session_id = data["session_id"]
    print(f"Session ID: {session_id}")
    
    # Add a dummy point at frame 0
    print("Adding prompt at frame 0")
    requests.post(f"{SAM_URL}/track/prompt", data={
        "session_id": session_id,
        "frame_idx": "0",
        "points": "[[500, 500]]",
        "point_labels": "[1]",
        "obj_id": "1"
    })
    
    # Propagate
    print("Starting propagation...")
    res = requests.post(f"{SAM_URL}/track/propagate", data={
        "session_id": session_id,
        "direction": "forward",
        "start_frame": "0",
        "max_frames": "-1"  # all frames
    }, stream=True)
    
    if res.status_code != 200:
        print(f"Propagate failed immediately: {res.status_code}")
        print(res.text)
        return

    # Read stream to keep connection alive and see progress
    for chunk in res.iter_lines():
        if chunk:
            print(chunk.decode())

if __name__ == "__main__":
    try:
        test_propagation()
    except Exception as e:
        print(f"Error: {e}")
