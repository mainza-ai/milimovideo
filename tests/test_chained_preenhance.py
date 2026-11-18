"""Chained generation must not re-enhance chunk 0 after the pipeline loads."""

from __future__ import annotations


def test_chunk_zero_enhancement_skips_when_preenhanced():
    from tasks.chained import should_enhance_chunk

    assert should_enhance_chunk({"enhance_prompt": True}, 0) is True
    assert should_enhance_chunk({"enhance_prompt": True, "_llm_preenhanced": True}, 0) is False
    assert should_enhance_chunk({"enhance_prompt": False}, 0) is False
    assert should_enhance_chunk({"enhance_prompt": True}, 1) is False
