import requests
import json
import base64
import os

API_URL = "http://localhost:8000"

def test_save_and_load():
    session_id = "test_verification_session"
    
    # create a dummy 1x1 pixel base64 image
    # Red pixel
    red_pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    
    # 1. Test Save
    print(f"Testing SAVE for session {session_id}...")
    save_payload = {
        "session_id": session_id,
        "frames": [
            {
                "frame_idx": 0,
                "masks": {"1": red_pixel},
                "scores": {"1": 0.95},
                "num_objects": 1
            },
             {
                "frame_idx": 5,
                "masks": {"1": red_pixel},
                "scores": {"1": 0.92},
                "num_objects": 1
            }
        ]
    }
    
    try:
        res = requests.post(f"{API_URL}/edit/track/save", json=save_payload)
        print(f"Save Status: {res.status_code}")
        print(f"Save Response: {res.text}")
        if res.status_code != 200:
            print("SAVE FAILED")
            return
    except Exception as e:
        print(f"Save Exception: {e}")
        return

    # 2. Test Load
    print(f"\nTesting LOAD for session {session_id}...")
    load_payload = {"session_id": session_id} # reusing TrackStopRequest schema which has session_id
    
    try:
        res = requests.post(f"{API_URL}/edit/track/load", json=load_payload)
        print(f"Load Status: {res.status_code}")
        if res.status_code == 200:
            data = res.json()
            print(f"Load Status: {data.get('status')}")
            frames = data.get('frames', [])
            print(f"Loaded Frames Count: {len(frames)}")
            if len(frames) == 2:
                print("SUCCESS: Loaded correct number of frames.")
                # Verify content
                f0 = next((f for f in frames if f['frameIndex'] == 0), None)
                if not f0 and 'frame_idx' in frames[0]: # handle backend returning frame_idx vs frameIndex
                     f0 = next((f for f in frames if f['frame_idx'] == 0), None)
                
                if f0:
                    print(f"Frame 0 masks found: {list(f0.get('masks', {}).keys())}")
                else:
                    print("Frame 0 not found in response!")
            else:
                print("FAILURE: Incorrect frame count.")
        else:
            print(f"LOAD FAILED: {res.text}")
            
    except Exception as e:
        print(f"Load Exception: {e}")

if __name__ == "__main__":
    test_save_and_load()
