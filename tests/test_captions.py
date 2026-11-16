from utils.captions import build_ass
from utils.render_utils import build_complex_filtergraph


def test_ass_has_styled_cues_and_escapes_text():
    ass = build_ass(
        [
            {"start_frame": 25, "end_frame": 75, "text": "Hello {world}"},
            {"start_frame": 75, "end_frame": 100, "text": "second line\nnext"},
            {"start_frame": 120, "end_frame": 150, "text": "   "},
        ],
        width=1920,
        height=1080,
        fps=25,
    )
    assert "PlayResX: 1920" in ass
    assert "PlayResY: 1080" in ass
    assert "Dialogue: 0,0:00:01.00,0:00:03.00" in ass
    assert r"Hello \{world\}" in ass
    assert r"second line\Nnext" in ass
    assert ass.count("Dialogue:") == 2


def test_filtergraph_burns_captions_after_compositing():
    cmd = build_complex_filtergraph(
        project_fps=25,
        project_width=320,
        project_height=180,
        v1_shots=[{
            "path": "/tmp/a.mp4", "start_frame": 0, "duration_frames": 25,
            "trim_in": 0, "gain": 1.0, "muted": False, "hidden": False,
            "has_audio": False,
        }],
        v2_shots=[],
        a1_shots=[],
        output_path="/tmp/out.mp4",
        captions_path="/tmp/captions.ass",
    )
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "ass=filename='/tmp/captions.ass'" in graph
    assert "[v_captioned]" in graph
    assert cmd[cmd.index("-map") + 1] == "[v_captioned]"
