from job_utils import _is_oom, _oom_retry_params


def test_oom_detection_matches_torch_and_mps_messages():
    assert _is_oom(MemoryError("boom"))
    assert _is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert _is_oom(RuntimeError("MPS backend out of memory"))
    assert not _is_oom(RuntimeError("model file missing"))


def test_oom_retry_lowers_resolution_and_frames():
    retry = _oom_retry_params("video", {
        "width": 1280, "height": 768, "num_frames": 121,
    })
    assert retry["width"] == 960
    assert retry["height"] == 576
    assert retry["num_frames"] == 89
    assert (retry["num_frames"] - 1) % 8 == 0

    image = _oom_retry_params("image", {"width": 512, "height": 512})
    assert image["width"] == 384
    assert image["height"] == 384
