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


def _cmd(v1=None, a1=None, track_states=None, **kwargs):
    return build_complex_filtergraph(
        project_fps=25,
        project_width=320,
        project_height=180,
        v1_shots=v1 or [],
        v2_shots=[],
        a1_shots=a1 or [],
        output_path="/tmp/out.mp4",
        track_states=track_states or {},
        **kwargs,
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


def test_crop_fit_covers_the_output_frame():
    cmd = _cmd(v1=[_clip()], fit="crop")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "force_original_aspect_ratio=increase" in graph
    assert "crop=320:180" in graph
    assert "pad=320:180" not in graph


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


def test_audio_normalization_and_bitrate_are_applied():
    cmd = _cmd(v1=[_clip()], a1=[_clip("/tmp/voice.wav")], audio_normalize=True)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "amix=inputs=2" in graph
    assert "loudnorm=I=-16:TP=-1.5:LRA=11[a_norm]" in graph
    maps = [cmd[index + 1] for index, value in enumerate(cmd) if value == "-map"]
    assert "[a_norm]" in maps

    off = _cmd(v1=[_clip()], a1=[_clip("/tmp/voice.wav")])
    off_graph = off[off.index("-filter_complex") + 1]
    assert "loudnorm" not in off_graph


def test_audio_bitrate_is_written_to_the_command():
    cmd = _cmd(v1=[_clip()], audio_bitrate="256k")
    assert "-b:a" in cmd
    assert cmd[cmd.index("-b:a") + 1] == "256k"


def test_titles_are_drawn_after_lut_and_escaped():
    cmd = _cmd(
        v1=[_clip()],
        lut_path="/tmp/grade.cube",
        titles=[{
            "text": "A:B's 100%", "start_frame": 0, "end_frame": 25,
            "x": 0.5, "y": 0.25, "size": 64, "color": "#ffcc00", "box": True,
        }],
    )
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "lut3d=" in graph
    assert "drawtext=" in graph
    assert "[v_lut]drawtext=" in graph
    assert "text='A\\:B\\'s 100\\%'" in graph
    assert "fontcolor=0xffcc00" in graph
    assert "x=(w-text_w)*0.5000" in graph
    assert "y=(h-text_h)*0.2500" in graph
    assert "enable='between(t,0.000,1.000)'" in graph
    assert graph.index("lut3d=") < graph.index("drawtext=")


def test_a2_mixes_with_a1_and_embedded_audio():
    cmd = _cmd(
        v1=[_clip()],
        a1=[
            _clip("/tmp/a1.wav", track_index=2),
            _clip("/tmp/a2.wav", track_index=3),
        ],
    )
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "amix=inputs=3" in graph
    assert "a_2" in graph and "a_3" in graph


def test_v1_transition_adds_an_alpha_ramp():
    cmd = _cmd(v1=[
        _clip(start_frame=0, duration_frames=50),
        _clip(start_frame=25, duration_frames=50, transition_in=25),
    ])
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "clip(T/1.0000,0,1)" in graph
    assert "alpha(X,Y)*" in graph
