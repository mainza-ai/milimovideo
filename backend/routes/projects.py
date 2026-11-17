from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from sqlmodel import Session, select, SQLModel
from database import get_session, Project, Shot, Scene
from schemas import (
    BoardLayoutPayload,
    BoardViewPayload,
    CreateProjectRequest,
    ProjectState,
    RenderRequest,
    RenderSettings,
    TimelineItem,
)
import uuid
import os
import shutil
import subprocess
import logging
import json
from datetime import datetime
import config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["projects"])


def _parse_json_field(value, default):
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _sanitize_keyframes(keyframes) -> list:
    points = {}
    for raw in list(keyframes or [])[:200]:
        data = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw or {})
        try:
            frame = max(0, int(data.get("frame") or 0))
            opacity = max(0.0, min(1.0, float(data.get("opacity", 1.0))))
            scale = max(0.1, min(4.0, float(data.get("scale", 1.0))))
            x = max(-1.0, min(1.0, float(data.get("x", 0.0))))
            y = max(-1.0, min(1.0, float(data.get("y", 0.0))))
            volume = max(0.0, min(2.0, float(data.get("volume", 1.0))))
        except (TypeError, ValueError):
            continue
        points[frame] = {
            "opacity": opacity,
            "scale": scale,
            "x": x,
            "y": y,
            "volume": volume,
        }  # last keyframe on a frame wins
    return [
        {"frame": frame, **values}
        for frame, values in sorted(points.items())
    ]


def _sanitize_blocking(nodes) -> list:
    cleaned = []
    for raw in list(nodes or [])[:100]:
        data = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw or {})
        node_type = str(data.get("type") or "character")
        if node_type not in {"camera", "character", "object"}:
            node_type = "character"
        try:
            x = max(0.0, min(100.0, float(data.get("x", 50))))
            y = max(0.0, min(100.0, float(data.get("y", 50))))
            rotation = float(data.get("rotation", 0))
        except (TypeError, ValueError):
            continue
        cleaned.append({
            "id": str(data.get("id") or uuid.uuid4().hex)[:40],
            "type": node_type,
            "label": str(data.get("label") or node_type.title())[:80],
            "x": x,
            "y": y,
            "rotation": max(-180.0, min(180.0, rotation)),
        })
    return cleaned


def _sanitize_captions(captions) -> list:
    cleaned = []
    for cue in list(captions or [])[:500]:
        raw = cue.model_dump() if hasattr(cue, "model_dump") else dict(cue or {})
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        start = max(0, int(raw.get("start_frame") or 0))
        end = max(start + 1, int(raw.get("end_frame") or 0))
        cleaned.append({
            "id": str(raw.get("id") or uuid.uuid4().hex)[:40],
            "start_frame": start,
            "end_frame": end,
            "text": text[:300],
        })
    cleaned.sort(key=lambda cue: cue["start_frame"])
    return cleaned


def _shot_dict(shot: Shot) -> dict:
    return {
        "id": shot.id,
        "project_id": shot.project_id,
        "scene_id": shot.scene_id,
        "index": shot.index,
        "sequence_index": shot.sequence_index,
        "prompt": shot.prompt,
        "width": shot.width,
        "height": shot.height,
        "seed": shot.seed,
        "seed_locked": shot.seed_locked,
        "num_frames": shot.num_frames,
        "fps": shot.fps,
        "cfg_scale": shot.cfg_scale,
        "negative_prompt": shot.negative_prompt,
        "enhance_prompt": shot.enhance_prompt,
        "upscale": shot.upscale,
        "pipeline_override": shot.pipeline_override,
        "auto_continue": shot.auto_continue,
        "status": shot.status,
        "thumbnail_url": shot.thumbnail_url,
        "video_url": shot.video_url,
        "track_index": shot.track_index,
        "start_frame": shot.start_frame,
        "trim_in": shot.trim_in,
        "trim_out": shot.trim_out,
        "action": shot.action,
        "dialogue": shot.dialogue,
        "character": shot.character,
        "shot_type": shot.shot_type,
        "timeline": _parse_json_field(shot.timeline, []),
        "keyframes": _parse_json_field(shot.keyframes, []),
        "blocking": _parse_json_field(shot.blocking, []),
        "matched_elements": _parse_json_field(shot.matched_elements, None),
        "enhanced_prompt_result": shot.enhanced_prompt_result,
        "last_job_id": shot.last_job_id,
        "selected_take_id": shot.selected_take_id,
        "favorite_take_id": shot.favorite_take_id,
        "audio_gain": shot.audio_gain,
        "audio_muted": shot.audio_muted,
        "speed": shot.speed,
        "audio_fade_in": shot.audio_fade_in,
        "audio_fade_out": shot.audio_fade_out,
        "created_at": shot.created_at.isoformat() if shot.created_at else None,
    }

@router.post("/projects", response_model=ProjectState)
def create_project(req: CreateProjectRequest, session: Session = Depends(get_session)):
    project_id = str(uuid.uuid4())[:8]
    
    # Create directory structure
    project_dir = os.path.join(config.PROJECTS_DIR, project_id)
    os.makedirs(os.path.join(project_dir, "generated"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "thumbnails"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "assets"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "workspace"), exist_ok=True) # For intermediate files
    
    new_project = Project(
        id=project_id,
        name=req.name,
        resolution_w=req.resolution_w,
        resolution_h=req.resolution_h,
        fps=req.fps,
        seed=req.seed
    )
    session.add(new_project)
    session.commit()
    session.refresh(new_project)
    
    # Return empty state
    return ProjectState(
        id=new_project.id,
        name=new_project.name,
        shots=[],
        resolution_w=new_project.resolution_w,
        resolution_h=new_project.resolution_h,
        fps=new_project.fps,
        seed=new_project.seed,
        created_at=new_project.created_at.timestamp() if new_project.created_at else None,
        updated_at=new_project.updated_at.timestamp() if new_project.updated_at else None,
        revision=new_project.revision,
    )

@router.get("/projects", response_model=List[ProjectState])
def get_projects(session: Session = Depends(get_session)):
    projects = session.exec(select(Project)).all()
    results = []
    for p in projects:
        # Get shot count or details? For list, maybe just summary, 
        # but ProjectState definition expects shots list.
        # We can optimize later, for now load shots.
        # Ensure we return valid shots structure
        results.append(ProjectState(
            id=p.id,
            name=p.name,
            shots=[], # Optimization: Don't load all shots for index
            resolution_w=p.resolution_w,
            resolution_h=p.resolution_h,
            fps=p.fps,
            seed=p.seed,
            created_at=p.created_at.timestamp() if p.created_at else None,
            updated_at=p.updated_at.timestamp() if p.updated_at else None,
            revision=p.revision,
        ))
    return results

@router.get("/projects/{project_id}", response_model=ProjectState)
def get_project(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Load shots
    # Load scenes
    scenes = session.exec(select(Scene).where(Scene.project_id == project_id).order_by(Scene.index)).all()
    scenes_data = []
    for sc in scenes:
        scenes_data.append({
            "id": sc.id,
            "index": sc.index,
            "name": sc.name,
            "script_content": sc.script_content
        })

    shots = session.exec(
        select(Shot)
        .where(Shot.project_id == project_id)
        .order_by(Shot.sequence_index.is_(None), Shot.sequence_index, Shot.created_at, Shot.id)
    ).all()

    return ProjectState(
        id=project.id,
        name=project.name,
        shots=[_shot_dict(s) for s in shots],
        scenes=scenes_data,
        captions=_parse_json_field(project.captions, []),
        captions_burn=bool(project.captions_burn),
        render_preset=project.render_preset,
        render_crf=project.render_crf or 18,
        render_speed=project.render_speed or "fast",
        render_audio_bitrate=project.render_audio_bitrate or "192k",
        render_audio_normalize=bool(project.render_audio_normalize),
        script_content=project.script_content,
        resolution_w=project.resolution_w,
        resolution_h=project.resolution_h,
        fps=project.fps,
        seed=project.seed,
        created_at=project.created_at.timestamp() if project.created_at else None,
        updated_at=project.updated_at.timestamp() if project.updated_at else None,
        revision=project.revision,
    )

@router.delete("/projects/{project_id}")
def delete_project(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
         raise HTTPException(status_code=404, detail="Project not found")
         
    # 1. Delete DB Records
    # Cascade delete shots? SQLModel might not handle cascade automatically unless configured
    shots = session.exec(select(Shot).where(Shot.project_id == project_id)).all()
    for s in shots:
        session.delete(s)
    scenes = session.exec(select(Scene).where(Scene.project_id == project_id)).all()
    for sc in scenes:
        session.delete(sc)
    session.delete(project)
    session.commit()
    
    # 2. Delete Filesystem
    project_dir = os.path.join(config.PROJECTS_DIR, project_id)
    if os.path.exists(project_dir):
        try:
            shutil.rmtree(project_dir)
            logger.info(f"Deleted project directory: {project_dir}")
        except Exception as e:
            logger.error(f"Failed to delete project directory {project_dir}: {e}")
    
    return {"status": "deleted"}

@router.put("/projects/{project_id}")
async def save_project(project_id: str, state: ProjectState, session: Session = Depends(get_session)):
    if state.id != project_id:
        raise HTTPException(status_code=400, detail="Project ID mismatch")
    
    db_project = session.get(Project, project_id)
    if not db_project:
        # UPSERT logic: If project doesn't exist (e.g., cleared DB but frontend still has default UI state), recreate it.
        logger.info(f"Project {project_id} not found, recreating...")
        db_project = Project(
            id=project_id,
            name=state.name,
            resolution_w=state.resolution_w,
            resolution_h=state.resolution_h,
            fps=state.fps,
            seed=state.seed
        )
        session.add(db_project)
    elif state.revision is not None and state.revision != db_project.revision:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Project revision conflict: client has {state.revision}, "
                f"server has {db_project.revision}. Reload before saving."
            ),
        )

    db_project.name = state.name
    
    # Save settings
    db_project.resolution_w = state.resolution_w
    db_project.resolution_h = state.resolution_h
    db_project.fps = state.fps
    db_project.seed = state.seed
    db_project.captions = json.dumps(_sanitize_captions(state.captions))
    db_project.captions_burn = bool(state.captions_burn)
    db_project.render_preset = state.render_preset
    db_project.render_crf = max(0, min(51, state.render_crf))
    db_project.render_speed = state.render_speed if state.render_speed in {
        "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow",
    } else "fast"
    db_project.render_audio_bitrate = state.render_audio_bitrate
    db_project.render_audio_normalize = bool(state.render_audio_normalize)
    db_project.script_content = state.script_content
    
    db_project.updated_at = datetime.utcnow()
    db_project.revision = (db_project.revision or 0) + 1

    # Sync Shots (Smart Merge)
    # Sync Scenes
    existing_scenes = session.exec(select(Scene).where(Scene.project_id == project_id)).all()
    existing_scene_map = {sc.id: sc for sc in existing_scenes}
    scene_payload_ids = set()
    scenes_to_add = []

    for sc_state in state.scenes:
        sc_data = sc_state.model_dump(exclude_unset=True)
        sc_id = sc_data.get("id")
        if not sc_id:
             sc_id = uuid.uuid4().hex
             sc_data["id"] = sc_id
        
        scene_payload_ids.add(sc_id)
        
        if sc_id in existing_scene_map:
             existing_scene = existing_scene_map[sc_id]
             existing_scene.name = sc_data["name"]
             existing_scene.index = sc_data.get("index", 0)
             existing_scene.script_content = sc_data.get("script_content")
             session.add(existing_scene)
        else:
             new_scene = Scene(
                 id=sc_id,
                 project_id=project_id,
                 name=sc_data["name"],
                 index=sc_data.get("index", 0),
                 script_content=sc_data.get("script_content")
             )
             scenes_to_add.append(new_scene)
             
    # Delete removed scenes
    for sc in existing_scenes:
        if sc.id not in scene_payload_ids:
            session.delete(sc)
            
    for sc in scenes_to_add:
        session.add(sc)


    # Sync Shots (Smart Merge)
    valid_keys = Shot.model_fields.keys()
    
    # 1. Fetch Key Existing Data to Preserve
    # Map Shot ID -> Existing Shot DB Object
    existing_shots = session.exec(select(Shot).where(Shot.project_id == project_id)).all()
    existing_map = {s.id: s for s in existing_shots}
    
    # Track IDs in the new payload to identify deletions
    payload_ids = set()
    
    new_shots_to_add = []
    
    for order, shot_state in enumerate(state.shots):
        s_data = shot_state.model_dump(exclude_unset=True)
        shot_id = s_data.get("id")
        if not shot_id:
            shot_id = uuid.uuid4().hex
            s_data["id"] = shot_id
        payload_ids.add(shot_id)

        # Clean input data. The payload array is the canonical story order.
        clean_data = {k: v for k, v in s_data.items() if k in valid_keys}
        clean_data["project_id"] = project_id
        clean_data["sequence_index"] = order
        if "timeline" in clean_data:
            clean_data["timeline"] = json.dumps(
                TimelineItem.normalize(clean_data.get("timeline"))
            )
        if "matched_elements" in clean_data:
            matched = _parse_json_field(clean_data.get("matched_elements"), None)
            clean_data["matched_elements"] = json.dumps(matched) if matched is not None else None
        if "keyframes" in clean_data:
            clean_data["keyframes"] = json.dumps(
                _sanitize_keyframes(clean_data.get("keyframes"))
            )
        if "blocking" in clean_data:
            clean_data["blocking"] = json.dumps(
                _sanitize_blocking(clean_data.get("blocking"))
            )

        if shot_id and shot_id in existing_map:
            # UPDATE Existing Shot
            existing_shot = existing_map[shot_id]

            for k, v in clean_data.items():
                if k in ["timeline", "matched_elements"]:
                    setattr(existing_shot, k, v)
                else:
                    setattr(existing_shot, k, v)

            session.add(existing_shot)
        else:
            # CREATE New Shot
            if not clean_data.get("id"):
                clean_data["id"] = uuid.uuid4().hex

            new_shots_to_add.append(Shot(**clean_data))
            
    # 2. DELETE output shots that are not in payload
    # (Unless we want to archive them? For now, standard sync means delete removed items)
    for s in existing_shots:
        if s.id not in payload_ids:
            session.delete(s)
            
    # 3. Add New
    for s in new_shots_to_add:
        session.add(s)
        
    session.add(db_project)
    session.commit()
    session.refresh(db_project)

    return {"status": "saved", "revision": db_project.revision}

@router.patch("/projects/{project_id}/settings")
async def update_project_settings(project_id: str, updates: dict, session: Session = Depends(get_session)):
    """Update project settings (resolution, fps, seed, name) after creation."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    allowed_fields = {"name", "resolution_w", "resolution_h", "fps", "seed"}
    applied = {}
    for key, value in updates.items():
        if key in allowed_fields and value is not None:
            setattr(project, key, value)
            applied[key] = value
    
    if not applied:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    
    session.add(project)
    session.commit()
    session.refresh(project)
    
    return {
        "status": "updated",
        "settings": {
            "name": project.name,
            "resolution_w": project.resolution_w,
            "resolution_h": project.resolution_h,
            "fps": project.fps,
            "seed": project.seed
        }
    }

async def _has_audio_stream(path: str) -> bool:
    import asyncio
    import subprocess
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=index", "-of", "csv=p=0", path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        return bool(stdout.strip())
    except Exception:
        return False


@router.get("/projects/{project_id}/integrity")
def project_integrity(project_id: str, session: Session = Depends(get_session)):
    """Report orphaned data, missing media, and ordering problems without mutating."""
    from file_utils import resolve_path

    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    shots = session.exec(select(Shot).where(Shot.project_id == project_id)).all()
    scenes = session.exec(select(Scene).where(Scene.project_id == project_id)).all()
    shot_ids = {shot.id for shot in shots}
    scene_ids = {scene.id for scene in scenes}
    issues = []

    for shot in shots:
        for field, label in (("video_url", "video"), ("thumbnail_url", "thumbnail")):
            value = getattr(shot, field)
            if not value:
                continue
            resolved = resolve_path(value)
            if not resolved or not os.path.exists(resolved):
                issues.append({
                    "kind": f"missing_{label}",
                    "severity": "error",
                    "shot_id": shot.id,
                    "path": value,
                    "message": f"Missing {label} for shot {shot.id}",
                })
        if shot.scene_id and shot.scene_id not in scene_ids:
            issues.append({
                "kind": "orphan_shot_scene",
                "severity": "warning",
                "shot_id": shot.id,
                "message": f"Shot {shot.id} references a missing scene",
            })
        if shot.sequence_index is None:
            issues.append({
                "kind": "missing_sequence_index",
                "severity": "warning",
                "shot_id": shot.id,
                "message": f"Shot {shot.id} has no sequence index",
            })

    for scene in scenes:
        if not any(shot.scene_id == scene.id for shot in shots):
            issues.append({
                "kind": "empty_scene",
                "severity": "info",
                "message": f"Scene {scene.id} has no shots",
            })

    layout = _parse_json_field(project.board_layout, {})
    for node in layout.get("nodes", []) if isinstance(layout, dict) else []:
        if node.get("id") not in shot_ids:
            issues.append({
                "kind": "orphan_board_node",
                "severity": "warning",
                "message": f"Board node {node.get('id')} has no shot",
            })
    for link in layout.get("links", []) if isinstance(layout, dict) else []:
        if link.get("source") not in shot_ids or link.get("target") not in shot_ids:
            issues.append({
                "kind": "orphan_board_link",
                "severity": "warning",
                "message": f"Link {link.get('id')} points at a missing shot",
            })

    for cue in _parse_json_field(project.captions, []):
        if cue.get("end_frame", 0) <= cue.get("start_frame", 0):
            issues.append({
                "kind": "invalid_caption",
                "severity": "warning",
                "message": f"Caption {cue.get('id')} has an empty time range",
            })

    return {
        "project_id": project_id,
        "ok": not any(issue["severity"] == "error" for issue in issues),
        "counts": {
            "shots": len(shots),
            "scenes": len(scenes),
            "issues": len(issues),
            "errors": sum(1 for issue in issues if issue["severity"] == "error"),
            "warnings": sum(1 for issue in issues if issue["severity"] == "warning"),
        },
        "issues": issues,
    }


@router.post("/projects/{project_id}/repair")
def repair_project(project_id: str, session: Session = Depends(get_session)):
    """Repair safe, reversible inconsistencies without deleting files.

    Clears missing media references, orphan scene/board references, normalizes
    sequence order, and sanitizes caption ranges. Missing files stay on disk for
    a manual relink; only the broken pointer is removed.
    """
    from file_utils import resolve_path

    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    shots = session.exec(select(Shot).where(Shot.project_id == project_id)).all()
    scenes = session.exec(select(Scene).where(Scene.project_id == project_id)).all()
    scene_ids = {scene.id for scene in scenes}
    shot_ids = {shot.id for shot in shots}
    counts = {
        "missing_video": 0,
        "missing_thumbnail": 0,
        "orphan_scene": 0,
        "sequence_reindexed": 0,
        "orphan_board_nodes": 0,
        "orphan_board_links": 0,
        "captions_sanitized": 0,
    }

    ordered = sorted(
        shots,
        key=lambda shot: (
            shot.sequence_index is None,
            shot.sequence_index or 0,
            shot.created_at,
            shot.id,
        ),
    )
    for index, shot in enumerate(ordered):
        if shot.sequence_index != index:
            shot.sequence_index = index
            counts["sequence_reindexed"] += 1
        if shot.scene_id and shot.scene_id not in scene_ids:
            shot.scene_id = None
            counts["orphan_scene"] += 1
        for field, key in (("video_url", "missing_video"), ("thumbnail_url", "missing_thumbnail")):
            value = getattr(shot, field)
            if not value:
                continue
            resolved = resolve_path(value)
            if not resolved or not os.path.exists(resolved):
                setattr(shot, field, None)
                counts[key] += 1
        session.add(shot)

    layout = _parse_json_field(project.board_layout, None)
    if isinstance(layout, dict):
        nodes = layout.get("nodes", [])
        links = layout.get("links", [])
        kept_nodes = [node for node in nodes if node.get("id") in shot_ids]
        kept_links = [
            link for link in links
            if link.get("source") in shot_ids and link.get("target") in shot_ids
        ]
        counts["orphan_board_nodes"] = len(nodes) - len(kept_nodes)
        counts["orphan_board_links"] = len(links) - len(kept_links)
        if counts["orphan_board_nodes"] or counts["orphan_board_links"]:
            layout["nodes"] = kept_nodes
            layout["links"] = kept_links
            project.board_layout = json.dumps(layout)
            project.board_revision = (project.board_revision or 0) + 1

    original_captions = _parse_json_field(project.captions, [])
    sanitized = _sanitize_captions(original_captions)
    if sanitized != original_captions:
        counts["captions_sanitized"] = max(1, len(original_captions) - len(sanitized))
        project.captions = json.dumps(sanitized)

    project.updated_at = datetime.utcnow()
    project.revision = (project.revision or 0) + 1
    session.add(project)
    session.commit()

    return {
        "project_id": project_id,
        "revision": project.revision,
        "counts": counts,
    }


@router.get("/projects/{project_id}/takes")
def list_project_takes(
    project_id: str,
    limit: int = 200,
    session: Session = Depends(get_session),
):
    """Every completed output in the project, with its shot and lineage."""
    from database import Job

    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    jobs = session.exec(
        select(Job)
        .where(Job.project_id == project_id)
        .where(Job.status == "completed")
        .where(Job.output_path.isnot(None))
        .where(Job.type.in_(["shot_generation", "advanced", "video", "image", "chained", "inpaint"]))
        .order_by(Job.created_at.desc())
        .limit(max(1, min(1000, limit)))
    ).all()

    shots = session.exec(select(Shot).where(Shot.project_id == project_id)).all()
    shot_map = {shot.id: shot for shot in shots}

    takes = []
    for job in jobs:
        shot = shot_map.get(job.shot_id or "")
        params = _parse_json_field(job.params_json, {})
        takes.append({
            "job_id": job.id,
            "shot_id": job.shot_id,
            "shot_label": (shot.action or shot.prompt or shot.id) if shot else None,
            "sequence_index": shot.sequence_index if shot else None,
            "url": job.output_path,
            "thumbnail_url": job.thumbnail_path,
            "type": job.type,
            "prompt": job.prompt,
            "enhanced_prompt": job.enhanced_prompt,
            "seed": params.get("seed") if isinstance(params, dict) else None,
            "model": (
                params.get("pipeline_override") or params.get("pipeline_type")
                if isinstance(params, dict) else None
            ),
            "num_frames": params.get("num_frames") if isinstance(params, dict) else None,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "selected": bool(shot and (shot.selected_take_id or shot.last_job_id) == job.id),
            "favorite": bool(shot and shot.favorite_take_id == job.id),
        })
    return {"takes": takes, "count": len(takes)}


@router.get("/projects/{project_id}/board")
def get_board(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    layout = _parse_json_field(project.board_layout, {"nodes": [], "areas": [], "links": []})
    return {"layout": layout, "revision": project.board_revision or 0}


@router.put("/projects/{project_id}/board")
def save_board(
    project_id: str,
    payload: BoardLayoutPayload,
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if payload.revision is not None and payload.revision != (project.board_revision or 0):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Board revision conflict: client has {payload.revision}, "
                f"server has {project.board_revision}."
            ),
        )
    layout = payload.model_dump(exclude={"revision"})
    project.board_layout = json.dumps(layout)
    project.board_revision = (project.board_revision or 0) + 1
    project.updated_at = datetime.utcnow()
    session.add(project)
    session.commit()
    return {"status": "saved", "revision": project.board_revision, "layout": layout}


@router.get("/projects/{project_id}/view")
def get_view(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"view": _parse_json_field(project.view_state, {})}


@router.put("/projects/{project_id}/view")
def save_view(
    project_id: str,
    payload: BoardViewPayload,
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    project.view_state = json.dumps(payload.model_dump())
    session.add(project)
    session.commit()
    return {"status": "saved"}


@router.post("/projects/{project_id}/render")
async def render_project(
    project_id: str,
    background_tasks: BackgroundTasks,
    req: Optional[RenderRequest] = None,
    session: Session = Depends(get_session),
):
    """Start a durable render job for the project's computed timeline."""
    from database import Job

    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    render_id = f"render_{uuid.uuid4().hex[:8]}"
    settings = (req.settings if req else RenderSettings()) or RenderSettings()
    track_states = (req.track_states if req else {}) or {}
    source_path = req.source_path if req else None
    if source_path:
        from file_utils import resolve_path
        resolved_source = resolve_path(source_path)
        if not resolved_source or not os.path.exists(resolved_source):
            raise HTTPException(status_code=404, detail="Render source not found")
    fps = settings.fps or project.fps
    width = settings.width or project.resolution_w
    height = settings.height or project.resolution_h

    job = Job(
        id=render_id,
        project_id=project_id,
        type="render",
        status="processing",
        progress=0,
        status_message="Initializing render...",
        params_json=json.dumps({
            "settings": settings.model_dump(),
            "track_states": track_states,
            "source_path": source_path,
        }),
    )
    session.add(job)
    session.commit()

    background_tasks.add_task(
        _render_background,
        project_id, render_id, fps, width, height, settings, track_states, source_path,
    )
    return {"job_id": render_id, "status": "rendering"}


async def _render_background(
    project_id: str,
    render_id: str,
    fps: int,
    width: int,
    height: int,
    settings: RenderSettings,
    track_states: dict,
    source_path: Optional[str] = None,
):
    """Build the filtergraph from computed layout, render, and stream progress."""
    import asyncio
    from events import event_manager
    from utils.render_utils import build_complex_filtergraph
    from utils.captions import cleanup as cleanup_captions, write_ass_file
    from file_utils import resolve_path
    from database import engine, Job
    from job_utils import active_jobs

    captions_path = None

    active_jobs[render_id] = {
        "cancelled": False,
        "status": "processing",
        "progress": 0,
        "status_message": "Initializing render...",
        "project_id": project_id,
        "type": "render",
    }

    with Session(engine) as session:
        job = session.get(Job, render_id)
        if not job:
            logger.error(f"Render Job {render_id} missing in background thread.")
            active_jobs.pop(render_id, None)
            return

        project_dir = os.path.join(config.PROJECTS_DIR, project_id, "generated")
        os.makedirs(project_dir, exist_ok=True)
        output_path = os.path.join(project_dir, f"{render_id}.mp4")
        video_url = f"/projects/{project_id}/generated/{render_id}.mp4"

        async def emit_progress(pct: int, message: str, status: str = "processing"):
            pct = max(0, min(100, int(pct)))
            if render_id in active_jobs:
                active_jobs[render_id].update({
                    "progress": pct, "status": status if status != "processing" else "processing",
                    "status_message": message,
                })
            job.progress = pct
            job.status_message = message
            job.status = "processing" if status == "processing" else status
            if status == "completed":
                job.output_path = video_url
                job.completed_at = datetime.utcnow()
            elif status in ("failed", "cancelled"):
                job.error_message = message if status == "failed" else None
                job.completed_at = datetime.utcnow()
            session.add(job)
            session.commit()
            session.refresh(job)

            payload = {
                "job_id": render_id,
                "project_id": project_id,
                "progress": pct,
                "message": message,
                "status": status,
            }
            if status == "completed":
                payload["video_url"] = video_url
            elif status == "failed":
                payload["error"] = message
            await event_manager.broadcast(
                "render_progress" if status == "processing" else f"render_{status}",
                payload,
            )

        try:
            await emit_progress(2, "Gathering computed timeline...")

            all_shots = session.exec(
                select(Shot)
                .where(Shot.project_id == project_id)
                .order_by(Shot.sequence_index.is_(None), Shot.sequence_index, Shot.created_at, Shot.id)
            ).all()

            v1_clips, v2_clips, a1_clips = [], [], []
            v1_cursor = 0

            if source_path:
                resolved_source = resolve_path(source_path)
                probe = (await asyncio.to_thread(
                    subprocess.check_output,
                    [
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(resolved_source),
                    ],
                    text=True,
                )).strip()
                source_frames = max(1, round(float(probe or 0) * fps))
                v1_clips = [{
                    "path": resolved_source,
                    "start_frame": 0,
                    "duration_frames": source_frames,
                    "source_frames": source_frames,
                    "speed": 1.0,
                    "fade_in": 0,
                    "fade_out": 0,
                    "trim_in": 0,
                    "keyframes": [],
                    "gain": 1.0,
                    "muted": False,
                    "hidden": False,
                    "has_audio": await _has_audio_stream(resolved_source),
                }]
            else:
                for shot in all_shots:
                    if not shot.video_url:
                        continue
                    resolved = resolve_path(shot.video_url)
                    if not resolved or not os.path.exists(resolved):
                        logger.warning(f"Render skipping missing media: {shot.video_url}")
                        continue

                    source_frames = max(
                        0,
                        (shot.num_frames or 121) - (shot.trim_in or 0) - (shot.trim_out or 0),
                    )
                    if source_frames <= 0:
                        continue
                    speed = min(4.0, max(0.25, shot.speed or 1.0))
                    duration_frames = max(1, round(source_frames / speed))

                    if shot.track_index == 0:
                        start_frame = v1_cursor
                        v1_cursor += duration_frames
                    else:
                        start_frame = shot.start_frame or 0

                    clip = {
                        "path": resolved,
                        "start_frame": start_frame,
                        "duration_frames": duration_frames,
                        "source_frames": source_frames,
                        "speed": speed,
                        "fade_in": shot.audio_fade_in or 0,
                        "fade_out": shot.audio_fade_out or 0,
                        "trim_in": shot.trim_in or 0,
                        "keyframes": _parse_json_field(shot.keyframes, []),
                        "gain": shot.audio_gain if shot.audio_gain is not None else 1.0,
                        "muted": bool(shot.audio_muted),
                        "hidden": False,
                        "has_audio": await _has_audio_stream(resolved),
                    }
                    if shot.track_index == 0:
                        v1_clips.append(clip)
                    elif shot.track_index == 1:
                        v2_clips.append(clip)
                    elif shot.track_index == 2:
                        a1_clips.append(clip)

            if not v1_clips and not v2_clips and not a1_clips:
                raise ValueError("No valid video or audio clips found to render.")

            stored_project = session.get(Project, project_id)
            captions = _parse_json_field(stored_project.captions, []) if stored_project else []
            if stored_project is not None and not stored_project.captions_burn:
                captions = []
            captions_path = write_ass_file(captions, width, height, fps)
            if captions_path:
                await emit_progress(4, f"Burning in {len(captions)} caption cue(s)...")

            total_frames = max(
                (clip["start_frame"] + clip["duration_frames"])
                for clip in (v1_clips + v2_clips + a1_clips)
            )
            range_start = max(0, min(total_frames - 1, settings.start_frame or 0))
            range_end = max(range_start + 1, min(total_frames, settings.end_frame or total_frames))
            range_duration = (range_end - range_start) / max(1, fps)

            await emit_progress(5, "Compiling FFmpeg filtergraph...")
            cmd = build_complex_filtergraph(
                project_fps=fps,
                project_width=width,
                project_height=height,
                v1_shots=v1_clips,
                v2_shots=v2_clips,
                a1_shots=a1_clips,
                output_path=output_path,
                track_states=track_states,
                crf=settings.crf,
                preset=settings.preset,
                audio_bitrate=settings.audio_bitrate,
                audio_normalize=settings.audio_normalize,
                captions_path=captions_path,
                start_seconds=range_start / max(1, fps),
                duration_seconds=range_duration,
                fit=settings.fit,
            )

            logger.info("Executing render FFmpeg: %s", " ".join(cmd))
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stderr_task = asyncio.create_task(process.stderr.read())

            duration_sec = max(0.05, range_duration)

            cancelled = False
            while True:
                if active_jobs.get(render_id, {}).get("cancelled"):
                    cancelled = True
                    process.terminate()
                    break
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", "ignore").strip()
                if text.startswith("out_time_us=") or text.startswith("out_time_ms="):
                    raw = text.split("=", 1)[1]
                    try:
                        # out_time_us is microseconds; out_time_ms is misleadingly
                        # microseconds in recent FFmpeg as well.
                        seconds = int(raw) / 1_000_000
                        pct = 5 + min(94, int((seconds / duration_sec) * 94))
                        await emit_progress(pct, f"Rendering {seconds:.1f}s / {duration_sec:.1f}s")
                    except ValueError:
                        pass

            returncode = await process.wait()
            stderr = (await stderr_task).decode("utf-8", "ignore")
            if cancelled:
                await emit_progress(0, "Render cancelled", status="cancelled")
                await event_manager.broadcast("cancelled", {"job_id": render_id})
                return
            if returncode != 0:
                raise RuntimeError(f"FFmpeg render failed: {stderr[-4000:]}")

            await emit_progress(100, "Render complete!", status="completed")
            logger.info(f"Render completed: {output_path}")

        except Exception as e:
            logger.error(f"Render failed: {e}")
            await emit_progress(0, str(e), status="failed")
        finally:
            cleanup_captions(captions_path)
            active_jobs.pop(render_id, None)


@router.get("/projects/{project_id}/renders")
async def get_project_renders(project_id: str, session: Session = Depends(get_session)):
    """Fetch history of completed project renders."""
    from database import Job
    statement = select(Job).where(Job.project_id == project_id, Job.type == "render").order_by(Job.created_at.desc())
    renders = session.exec(statement).all()
    return renders


@router.get("/projects/{project_id}/render/{render_id}/download")
async def download_render(project_id: str, render_id: str):
    """Serve the rendered video file as a download."""
    from fastapi.responses import FileResponse
    
    file_path = os.path.join(config.PROJECTS_DIR, project_id, "generated", f"{render_id}.mp4")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Render not found")
    
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=f"{render_id}.mp4"
    )

@router.get("/projects/{project_id}/images")
async def get_project_images(project_id: str, session: Session = Depends(get_session)):
    """List all generated images for a project."""
    from database import Asset
    statement = select(Asset).where(Asset.project_id == project_id, Asset.type == "image").order_by(Asset.created_at.desc())
    results = session.exec(statement).all()
    return results


class SplitShotRequest(SQLModel):
    split_frame: int # Local frame index in the shot

@router.post("/shots/{shot_id}/split")
async def split_shot(shot_id: str, req: SplitShotRequest, session: Session = Depends(get_session)):
    """
    Split a shot into two at the given local frame index.
    Adjusts trim_out of original and creates new shot with trim_in.
    """
    original_shot = session.get(Shot, shot_id)
    if not original_shot:
        raise HTTPException(status_code=404, detail="Shot not found")
        
    project = session.get(Project, original_shot.project_id)
    if not project:
         raise HTTPException(status_code=404, detail="Project not found")

    # 1. Validate Split Point
    # Must be > 0 and < (num_frames - trim_in - trim_out)
    current_duration = original_shot.num_frames - (original_shot.trim_in or 0) - (original_shot.trim_out or 0)
    
    if req.split_frame <= 0 or req.split_frame >= current_duration:
        raise HTTPException(status_code=400, detail="Split point out of bounds")
        
    # Absolute frame within the source video
    absolute_split_frame = (original_shot.trim_in or 0) + req.split_frame
    
    # 2. Update Original Shot (Trim Out)
    # The new trim_out should be such that shot ends at absolute_split_frame
    # num_frames - trim_out_new = absolute_split_frame
    # trim_out_new = num_frames - absolute_split_frame
    
    old_trim_out = original_shot.trim_out or 0
    new_trim_out = original_shot.num_frames - absolute_split_frame
    
    original_shot.trim_out = new_trim_out
    session.add(original_shot)
    
    # 3. Shift the project-wide story order and the scene-local index so the
    # new shot lands immediately after the original, not at the end.
    new_sequence_index = None
    if original_shot.sequence_index is not None:
        new_sequence_index = original_shot.sequence_index + 1
        for other in session.exec(
            select(Shot)
            .where(Shot.project_id == original_shot.project_id)
            .where(Shot.sequence_index >= new_sequence_index)
        ).all():
            other.sequence_index = (other.sequence_index or 0) + 1
            session.add(other)

    new_scene_index = None
    if original_shot.index is not None and original_shot.scene_id:
        new_scene_index = original_shot.index + 1
        for other in session.exec(
            select(Shot)
            .where(Shot.scene_id == original_shot.scene_id)
            .where(Shot.index >= new_scene_index)
        ).all():
            other.index = (other.index or 0) + 1
            session.add(other)

    # 4. Create New Shot (Copy of Original)
    new_shot_data = original_shot.model_dump(exclude={"id", "created_at"})
    new_shot = Shot(**new_shot_data)
    new_shot.id = uuid.uuid4().hex
    new_shot.created_at = datetime.utcnow()
    new_shot.sequence_index = new_sequence_index
    new_shot.index = new_scene_index
    new_shot.trim_in = absolute_split_frame
    new_shot.trim_out = old_trim_out

    # 5. Free tracks keep absolute position; V1 is magnetic from sequence_index.
    if original_shot.track_index != 0:
        new_shot.start_frame = (original_shot.start_frame or 0) + req.split_frame

    session.add(new_shot)
    session.commit()
    session.refresh(original_shot)
    session.refresh(new_shot)

    def _as_dict(shot: Shot) -> dict:
        data = shot.model_dump()
        for key in ("timeline", "matched_elements"):
            value = data.get(key)
            if isinstance(value, str):
                try:
                    data[key] = json.loads(value) if value else ([] if key == "timeline" else None)
                except json.JSONDecodeError:
                    data[key] = [] if key == "timeline" else None
        return data

    return {
        "original_shot": _as_dict(original_shot),
        "new_shot": _as_dict(new_shot),
    }


# ── LLM Settings ──────────────────────────────────────────────────

@router.get("/settings/llm")
def get_llm_settings():
    """Return current LLM configuration."""
    return {
        "provider": config.LLM_PROVIDER,
        "ollama_base_url": config.OLLAMA_BASE_URL,
        "ollama_model": config.OLLAMA_MODEL,
        "ollama_keep_alive": config.OLLAMA_KEEP_ALIVE,
    }


@router.patch("/settings/llm")
def update_llm_settings(body: dict):
    """Update LLM settings at runtime."""
    if "provider" in body:
        if body["provider"] not in ("gemma", "ollama"):
            raise HTTPException(400, f"Invalid provider: {body['provider']}. Must be 'gemma' or 'ollama'.")
        config.LLM_PROVIDER = body["provider"]
    if "ollama_model" in body:
        config.OLLAMA_MODEL = body["ollama_model"]
    if "ollama_base_url" in body:
        config.OLLAMA_BASE_URL = body["ollama_base_url"]
    if "ollama_keep_alive" in body:
        config.OLLAMA_KEEP_ALIVE = body["ollama_keep_alive"]
    
    logger.info(f"LLM settings updated: provider={config.LLM_PROVIDER}, model={config.OLLAMA_MODEL}, keep_alive={config.OLLAMA_KEEP_ALIVE}")
    return get_llm_settings()


# Vision model families — these can accept images via Ollama's multimodal API
VISION_FAMILIES = {"mllama", "clip", "llava", "bakllava", "moondream"}

@router.get("/settings/llm/models")
def list_ollama_models():
    """Query Ollama for available models, tagging vision-capable ones."""
    import requests
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        
        # Detect which models support vision (image input)
        result = []
        for m in models:
            families = set(m.get("details", {}).get("families") or [])
            is_vision = bool(families & VISION_FAMILIES)
            # Skip embedding-only models (bert, nomic-bert)
            family = m.get("details", {}).get("family", "")
            if family in ("bert", "nomic-bert"):
                continue
            result.append({
                "name": m["name"],
                "size": m.get("size"),
                "modified_at": m.get("modified_at"),
                "is_vision": is_vision,
                "parameter_size": m.get("details", {}).get("parameter_size"),
            })
        
        return {"models": result}
    except requests.ConnectionError:
        raise HTTPException(503, f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}")
    except Exception as e:
        raise HTTPException(500, str(e))
