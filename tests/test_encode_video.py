"""The CUDA encoder must consume the lazy video generator under no_grad.

The LTX pipeline creates the decoded frame generator inside
``torch.inference_mode()``. On CUDA, calling ``next()`` outside that context
with autograd active raises "Inference tensors cannot be saved for backward".
``encode_video`` now wraps consumption in ``torch.no_grad()``; this test fails
before that fix because the generator observes grad mode.
"""

from __future__ import annotations

import torch

from ltx_pipelines.utils.media_io import encode_video


def test_encode_video_consumes_the_generator_without_grad(tmp_path):
    def frames():
        for _ in range(2):
            assert not torch.is_grad_enabled(), "encoder consumed frames with autograd enabled"
            yield torch.zeros((1, 8, 8, 3), dtype=torch.uint8)

    output = tmp_path / "encoder-no-grad.mp4"
    encode_video(iter(frames()), 1, None, None, str(output), 2)

    assert output.exists()
    assert output.stat().st_size > 0
