"""
TrackingManager — manages SAM 3 video tracking sessions.
Proxies to the SAM microservice's /track/* endpoints.
"""
import os
import requests
import logging
import config

logger = logging.getLogger(__name__)

SAM_URL = f"http://localhost:{config.SAM_SERVICE_PORT}"


class TrackingManager:

    def start_session(self, video_path: str, session_id: str = None) -> dict:
        """Start a tracking session on a video file."""
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return {"error": f"Video not found: {video_path}"}

        try:
            data = {"video_path": video_path}
            if session_id:
                data["session_id"] = session_id

            res = requests.post(f"{SAM_URL}/track/start", data=data, timeout=60)
            if res.status_code == 200:
                return res.json()
            else:
                logger.error(f"Track start failed: {res.text}")
                return {"error": res.text}
        except Exception as e:
            logger.error(f"Track start connection error: {e}")
            return {"error": str(e)}

    def add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        text: str = None,
        points: list = None,
        point_labels: list = None,
        boxes: list = None,
        box_labels: list = None,
        obj_id: int = None,
    ) -> dict:
        """Add a prompt to a tracking session."""
        import json as json_mod
        try:
            data = {
                "session_id": session_id,
                "frame_idx": str(frame_idx),
            }
            if text:
                data["text"] = text
            if points:
                data["points"] = json_mod.dumps(points)
            if point_labels:
                data["point_labels"] = json_mod.dumps(point_labels)
            if boxes:
                data["boxes"] = json_mod.dumps(boxes)
            if box_labels:
                data["box_labels"] = json_mod.dumps(box_labels)
            if obj_id is not None:
                data["obj_id"] = str(obj_id)

            res = requests.post(f"{SAM_URL}/track/prompt", data=data, timeout=30)
            if res.status_code == 200:
                return res.json()
            else:
                logger.error(f"Track prompt failed: {res.text}")
                return {"error": res.text}
        except Exception as e:
            logger.error(f"Track prompt connection error: {e}")
            return {"error": str(e)}

    def propagate(
        self,
        session_id: str,
        direction: str = "forward",
        start_frame: int = 0,
        max_frames: int = -1,
    ) -> dict:
        """Propagate tracking through video frames."""
        try:
            data = {
                "session_id": session_id,
                "direction": direction,
                "start_frame": str(start_frame),
                "max_frames": str(max_frames),
            }
            res = requests.post(f"{SAM_URL}/track/propagate", data=data, timeout=300)
            if res.status_code == 200:
                return res.json()
            else:
                logger.error(f"Track propagate failed: {res.text}")
                return {"error": res.text}
        except Exception as e:
            logger.error(f"Track propagate connection error: {e}")
            return {"error": str(e)}

    def stop_session(self, session_id: str) -> dict:
        """Close a tracking session."""
        try:
            data = {"session_id": session_id}
            res = requests.post(f"{SAM_URL}/track/stop", data=data, timeout=10)
            if res.status_code == 200:
                return res.json()
            else:
                logger.error(f"Track stop failed: {res.text}")
                return {"error": res.text}
        except Exception as e:
            logger.error(f"Track stop connection error: {e}")
            return {"error": str(e)}


tracking_manager = TrackingManager()
