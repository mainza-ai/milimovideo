from utils.render_utils import build_complex_filtergraph


def _clip(path="/tmp/a.mp4", **overrides):
    clip = {
        "path": path,
        "start_frame": 0,
        "duration_frames": 25,
        "trim_in": 0,
        "gain": 1.0,
        "muted": False,
        "hidden": False,
        "has_audio": True,
    }
    clip.update(overrides)
    return clip


def _cmd(v1=None, a1=None, track_states=None):
    return build_complex_filtergraph(
        project_fps=25,
        project_width=320,
        project_height=180,
        v1_shots=v1 or [],
        v2_shots=[],
        a1_shots=a1 or [],
        output_path="/tmp/out.mp4",
        track_states=track_states or {},
    )


def test_clip_gain_is_written_into_the_audio_chain():
    cmd = _cmd(v1=[_clip(gain=0.5)])
    assert any("volume=0.5" in part for part in cmd)


def test_muted_clip_contributes_no_audio_chain_but_keeps_video():
    cmd = _cmd(v1=[_clip(muted=True)])
    assert not any("atrim" in part for part in cmd)
    assert any("[v1_0]" in part for part in cmd)


def test_muted_a1_track_is_skipped_and_track_state_mute_wins():
    cmd = _cmd(a1=[_clip("/tmp/voice.wav")], track_states={"2": {"muted": True}})
    assert not any("atrim" in part for part in cmd)

    audible = _cmd(a1=[_clip("/tmp/voice.wav")])
    assert any("atrim" in part for part in audible)


def test_hidden_v1_leaves_no_video_chain():
    cmd = _cmd(v1=[_clip()], track_states={"0": {"hidden": True}})
    assert not any("[v1_0]" in part for part in cmd)
