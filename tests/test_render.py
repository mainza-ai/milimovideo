from utils.render_utils import build_complex_filtergraph


def test_filtergraph_uses_v1_layout_v2_alpha_and_embedded_audio():
    v1 = [{
        "path": "/tmp/a.mp4", "start_frame": 0, "duration_frames": 100,
        "trim_in": 0, "has_audio": True, "gain": 1.0, "muted": False, "hidden": False,
    }]
    v2 = [{
        "path": "/tmp/b.mov", "start_frame": 25, "duration_frames": 50,
        "trim_in": 0, "has_audio": False, "gain": 1.0, "muted": False, "hidden": False,
    }]
    a1 = [{
        "path": "/tmp/music.wav", "start_frame": 0, "duration_frames": 200,
        "trim_in": 0, "has_audio": True, "gain": 0.8, "muted": False, "hidden": False,
    }]
    cmd = build_complex_filtergraph(
        project_fps=25,
        project_width=320,
        project_height=180,
        v1_shots=v1,
        v2_shots=v2,
        a1_shots=a1,
        output_path="/tmp/out.mp4",
    )
    joined = " ".join(cmd)
    assert "color=black@0" in joined
    assert "format=rgba" in joined
    assert "[1:a]" in joined
    assert "amix=inputs=2:duration=longest" in joined
    assert "volume=0.8" in joined
    assert "-t" in cmd


def test_filtergraph_respects_muted_and_hidden_tracks():
    clips = [{
        "path": "/tmp/a.mp4", "start_frame": 0, "duration_frames": 100,
        "trim_in": 0, "has_audio": True, "gain": 1.0, "muted": False, "hidden": False,
    }]
    cmd = build_complex_filtergraph(
        project_fps=25, project_width=320, project_height=180,
        v1_shots=clips, v2_shots=[], a1_shots=[],
        output_path="/tmp/out.mp4",
        track_states={"0": {"muted": True, "hidden": True}},
    )
    joined = " ".join(cmd)
    assert "[1:a]" not in joined
    assert "overlay" not in joined
