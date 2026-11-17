def test_cancel_an_active_download(app_client):
    from services.model_downloader import DownloadState, model_downloader

    state = DownloadState(model_id="depth-anything-v2-small", total_bytes=100)
    model_downloader.active_downloads["depth-anything-v2-small"] = state
    try:
        response = app_client.delete("/models/depth-anything-v2-small/download")
        assert response.status_code == 200, response.text
        assert state.cancelled is True
    finally:
        model_downloader.active_downloads.pop("depth-anything-v2-small", None)


def test_cancel_a_queued_download(app_client):
    from services.model_downloader import QueuedDownload, model_downloader

    queued = QueuedDownload(model_id="depth-anything-v2-small", name="Depth", size_bytes=100)
    model_downloader._queue.append(queued)
    try:
        response = app_client.delete("/models/depth-anything-v2-small/download")
        assert response.status_code == 200, response.text
        assert all(item.model_id != "depth-anything-v2-small" for item in model_downloader._queue)
    finally:
        model_downloader._queue[:] = [item for item in model_downloader._queue if item.model_id != "depth-anything-v2-small"]


def test_cancel_missing_download_returns_404(app_client):
    response = app_client.delete("/models/nope/download")
    assert response.status_code == 404
