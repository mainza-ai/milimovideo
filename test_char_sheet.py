import requests
import json
import time

project_id = "test_char_sheet_project"
img_path = "/Users/mck/Desktop/milimovideo/assets/milimo_video_elements.png"

print("Creating project...")
requests.post("http://localhost:8000/projects", json={
    "name": "Test Char Sheet",
    "project_id": project_id
})

print("Launching Image-to-Video job with a generic infographic as input...")
res = requests.post("http://localhost:8000/jobs", json={
    "command": "generate_video",
    "project_id": project_id,
    "params": {
        "prompt": "A highly detailed character turnaround sheet of a fantasy warrior in full armor, standing in a t-pose. The background is pure white.",
        "pipeline_type": "ti2vid",
        "num_frames": 49,
        "enhance_prompt": True,
        "input_images": [[img_path, 0, 1.0]],
    }
})

if res.status_code != 200:
    print(f"Error starting job: {res.text}")
    exit(1)

job_id = res.json()["job_id"]
print(f"Launched job {job_id}")

last_progress = -1
while True:
    try:
        status_res = requests.get(f"http://localhost:8000/status/{job_id}").json()
        prog = status_res.get("progress", 0)
        if prog != last_progress:
            print(f"Status: {status_res['status']} | Progress: {prog}% | {status_res.get('message', '')}")
            last_progress = prog
        
        if status_res["status"] in ["completed", "failed"]:
            print(f"Final prompt used (check if JSON parsed correctly!):")
            print(status_res.get("enhanced_prompt", ""))
            break
            
    except Exception as e:
        print("Waiting for server response...")
        
    time.sleep(2)
