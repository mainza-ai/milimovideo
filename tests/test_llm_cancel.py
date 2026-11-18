"""Non-stream HTTP cancellation aborts the socket instead of waiting for timeout."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class _SlowHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802 - http.server API
        time.sleep(3)
        body = b'{"choices":[{"message":{"content":"late"}}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.mark.asyncio
async def test_acomplete_cancels_an_inflight_non_stream_request():
    from services.llm import providers

    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        with pytest.raises(providers.ProviderCancelled):
            await providers.acomplete(
                provider="fake",
                wire="openai_compatible",
                base_url=f"http://127.0.0.1:{server.server_address[1]}",
                api_key=None,
                model="fake-model",
                messages=[providers.ChatMessage(role="user", content="hi")],
                options={},
                timeout_sec=30,
                cancel_check=lambda: True,
            )
        assert time.monotonic() - started < 2.0
    finally:
        server.shutdown()
        server.server_close()
