from utils.render_utils import build_complex_filtergraph, opacity_expression


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


def test_opacity_expression_holds_and_interpolates():
    assert opacity_expression([], 25) is None
    assert opacity_expression([{"frame": 5, "opacity": 0.4}], 25) == "0.4000"

    expr = opacity_expression(
        [{"frame": 0, "opacity": 0.0}, {"frame": 25, "opacity": 1.0}],
        25,
    )
    assert expr.startswith("if(lt(T,0.000000),0.0000,")
    assert "1.000000" in expr  # slope over one second


def test_audio_fades_are_written():
    cmd = _cmd(v1=[_clip(fade_in=0.5, fade_out=0.2)])
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "afade=t=in:st=0:d=0.500" in graph
    assert "afade=t=out:st=" in graph


def test_solo_mutes_other_audio_tracks():
    both = _cmd(v1=[_clip()], a1=[_clip("/tmp/voice.wav")])
    both_graph = both[both.index("-filter_complex") + 1]
    assert both_graph.count("atrim") == 2

    solo = _cmd(v1=[_clip()], a1=[_clip("/tmp/voice.wav")], track_states={"0": {"solo": True}})
    solo_graph = solo[solo.index("-filter_complex") + 1]
    assert solo_graph.count("atrim") == 1


def test_speed_scales_video_and_audio_time():
    cmd = _cmd(v1=[_clip(speed=2.0, source_frames=50, duration_frames=25)])
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "setpts=(PTS-STARTPTS)/2.0" in graph
    assert "atempo=2.0" in graph
    assert "trim=start=0.0:duration=2.0" in graph


def test_scale_pan_and_volume_keyframes():
    keyframes = [
        {"frame": 0, "opacity": 1, "scale": 1.0, "x": 0, "y": 0, "volume": 0.2},
        {"frame": 24, "opacity": 1, "scale": 1.3, "x": 0.2, "y": -0.1, "volume": 1.0},
    ]
    cmd = _cmd(v1=[_clip(keyframes=keyframes)])
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "zoompan=" in graph
    assert "volume=eval=frame" in graph
    assert "geq=" not in graph  # opacity is constant, so no alpha pass


def test_keyframes_add_alpha_expression_to_the_clip():
    cmd = _cmd(v1=[_clip(keyframes=[{"frame": 0, "opacity": 0.2}, {"frame": 24, "opacity": 1.0}])])
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "geq=" in graph
    assert "alpha(X,Y)" in graph
    assert "format=rgba" in graph
