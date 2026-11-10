import pytest
from starlette.testclient import TestClient
from pathlib import Path
from facecls.server import app, PROJECT_ROOT

client = TestClient(app)

def test_serve_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "Face Recognition Studio" in response.text

def test_api_stats():
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_persons" in data
    assert "engine" in data
    assert data["total_persons"] >= 0

def test_api_persons():
    response = client.get("/api/persons")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_api_unclassified():
    response = client.get("/api/unclassified")
    assert response.status_code == 200
    data = response.json()
    assert "files" in data
    assert "total" in data

def test_api_config():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "engine" in data
    assert "cosine_threshold" in data

def test_path_traversal_protection():
    response = client.get("/api/media/../../etc/passwd")
    assert response.status_code in (403, 404)

from unittest.mock import patch, MagicMock

def test_api_scan_dry_run(tmp_path):
    mock_engine = MagicMock()
    mock_engine.engine = "insightface"
    mock_engine.app.get.return_value = []
    with patch("facecls.server.load_known_faces", return_value=({}, None)), \
         patch("facecls.server.get_shared_engine", return_value=mock_engine), \
         patch("facecls.server.load_image_rgb", return_value=None):
        response = client.post("/api/scan", json={"apply": False})
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "results" in data
        assert data["applied"] is False

def test_cancel_scan_and_download():
    res_scan = client.post("/api/scan/cancel")
    assert res_scan.status_code == 200
    assert res_scan.json()["success"] is True

    res_dl = client.post("/api/download/cancel")
    assert res_dl.status_code == 200
    assert res_dl.json()["success"] is True

def test_media_head_request():
    # HEAD request must be allowed (status 200 or 404, not 405 Method Not Allowed)
    res = client.head("/api/media/dataset/sontung/sontung_01.jpg")
    assert res.status_code in (200, 404)

def test_scan_stream_endpoint():
    with patch("facecls.server.load_known_faces", return_value=({}, None)):
        with client.stream("GET", "/api/scan/stream?apply=false") as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data:"):
                    import json
                    payload = json.loads(line[5:])
                    assert payload["type"] in ("start", "item", "complete", "cancelled")
                    break

def test_stream_download_endpoint_accepts_modes():
    with patch("facecls.server.search_instagram_images", return_value=[]), \
         patch("facecls.server.search_image_urls_page", return_value=[]):
        with client.stream("GET", "/api/download/stream?query=test_user&mode=instagram&dest_mode=download_person&count=1") as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data:"):
                    import json
                    payload = json.loads(line[5:])
                    assert payload["type"] == "start"
                    assert "instagram" in payload.get("mode", "")
                    break

def test_proxy_image_endpoint():
    # Invalid url returns 404
    res = client.get("/api/proxy_image?url=http://invalid-url-that-does-not-exist.test/pic.jpg")
    assert res.status_code == 404

    # HEAD request is also supported
    res_head = client.head("/api/proxy_image?url=http://invalid-url-that-does-not-exist.test/pic.jpg")
    assert res_head.status_code == 404

def test_stream_download_clean_target():
    with patch("facecls.server.iter_instagram_images", return_value=[("http://example.com/avatar.jpg", True, "Avatar")]), \
         patch("facecls.server.download_and_verify_image", return_value=None):
        with client.stream("GET", "/api/download/stream?query=test_user&mode=instagram&dest_mode=download_person&count=1&clean_target=true") as response:
            assert response.status_code == 200
            types = []
            for line in response.iter_lines():
                if line.startswith("data:"):
                    import json
                    payload = json.loads(line[5:])
                    types.append(payload["type"])
                    if payload["type"] == "candidate_found":
                        assert payload["is_avatar"] is True
                        break
            assert "start" in types

def test_stream_download_all_mode():
    with patch("facecls.server.iter_instagram_images", return_value=[
        ("http://example.com/avatar.jpg", True, "Avatar"),
        ("http://example.com/post1.jpg", False, "Post 1"),
        ("http://example.com/post2.jpg", False, "Post 2"),
    ]), patch("facecls.server.download_and_verify_image", return_value=None):
        with client.stream("GET", "/api/download/stream?query=test_user&mode=instagram&dest_mode=download_person&download_all=true") as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data:"):
                    import json
                    payload = json.loads(line[5:])
                    if payload["type"] == "start":
                        assert payload.get("unlimited") is True
                        assert payload.get("count") == 0
                        break

def test_stream_download_facebook_mode():
    with patch("facecls.server.resolve_facebook_profile_info", return_value=("caca.tieu.50309", "Huỳnh Giang", "100050246662860", "http://example.com/fb_avatar.jpg")), \
         patch("facecls.server.iter_facebook_images", return_value=[
             ("http://example.com/fb_avatar.jpg", True, "Avatar Facebook"),
             ("http://example.com/fb_photo1.jpg", False, "Photo 1"),
         ]), patch("facecls.server.download_and_verify_image", return_value=None):
        with client.stream("GET", "/api/download/stream?query=https://www.facebook.com/caca.tieu.50309/photos&dest_mode=download_person&count=2") as response:
            assert response.status_code == 200
            events = []
            for line in response.iter_lines():
                if line.startswith("data:"):
                    import json
                    payload = json.loads(line[5:])
                    events.append(payload)
                    if payload["type"] == "candidate_found":
                        assert payload["source"] == "facebook"
                        break
            start_ev = next(e for e in events if e["type"] == "start")
            assert start_ev["mode"] == "facebook"
            assert "caca_tieu_50309" in start_ev["target"]


def test_save_candidates_endpoint():
    import shutil
    from io import BytesIO
    from PIL import Image
    import base64
    from facecls.server import PROJECT_ROOT

    img = Image.new("RGB", (10, 10), color="blue")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    test_dir = "tests_tmp_manual_saved"
    full_test_dir = PROJECT_ROOT / test_dir
    try:
        payload = {
            "target_dir": test_dir,
            "prefix": "test_person",
            "items": [
                {"id": 101, "url": b64_str, "desc": "Candidate 1"},
                {"id": 102, "url": "invalid_url_scheme_123", "desc": "Invalid Cand"}
            ]
        }
        res = client.post("/api/download/save_candidates", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["saved_count"] == 1
        assert data["failed_count"] == 1
        assert data["saved"][0]["id"] == 101
        assert (full_test_dir / "test_person_0001.jpg").exists()
    finally:
        if full_test_dir.exists():
            shutil.rmtree(full_test_dir, ignore_errors=True)


def test_save_candidates_security_check():
    payload = {
        "target_dir": "/etc/sensitive_dir",
        "items": [{"id": 1, "url": "data:image/jpeg;base64,xxx"}]
    }
    res = client.post("/api/download/save_candidates", json=payload)
    assert res.status_code == 403


def test_scan_video_handling(tmp_path):
    from unittest.mock import patch, MagicMock
    import numpy as np
    from facecls.config import AppConfig

    test_unclass = tmp_path / "test_unclass_video"
    test_unclass.mkdir()
    fake_vid = test_unclass / "sample_clip.mp4"
    fake_vid.write_bytes(b"dummy mp4 binary")

    cfg = AppConfig()
    cfg.dataset_root = str(tmp_path)
    cfg.unclassified_dir = test_unclass.name

    mock_engine = MagicMock()
    mock_engine.extract_frames_from_video.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)]
    
    mock_index = MagicMock()
    mock_index.match_video.return_value = (
        "sontung",
        "video: 1/1 frames khớp sontung (conf TB 92.5%)",
        92.5,
        {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4, "score": 92.5}
    )

    with patch("facecls.server.load_config", return_value=cfg), \
         patch("facecls.server.get_shared_engine", return_value=mock_engine), \
         patch("facecls.server.load_known_faces", return_value=({"sontung": []}, mock_index)):
        res = client.post("/api/scan", json={"force": True})
        assert res.status_code == 200
        data = res.json()
        assert data["stats"]["total"] == 1
        assert data["stats"]["recognized"] == 1
        assert len(data["results"]) == 1
        item = data["results"][0]
        assert item["person"] == "sontung"
        assert item["confidence"] == 92.5
        assert item["is_video"] is True
        assert item["bbox"] is not None


def test_playground_video_upload():
    from unittest.mock import patch, MagicMock
    import numpy as np
    from io import BytesIO

    mock_engine = MagicMock()
    mock_engine.extract_frames_from_video.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)]

    mock_index = MagicMock()
    mock_index.match_video.return_value = (
        "sontung",
        "video: 1/1 frames khớp sontung (conf TB 90.0%)",
        90.0,
        {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5, "score": 90.0}
    )

    fake_video_bytes = b"fake mp4 video bytes"
    files = {"file": ("test_clip.mp4", BytesIO(fake_video_bytes), "video/mp4")}

    with patch("facecls.server.get_shared_engine", return_value=mock_engine), \
         patch("facecls.server.load_known_faces", return_value=({"sontung": []}, mock_index)), \
         patch("facecls.server.check_sensitive_media", return_value={"is_sensitive": False, "detections": []}):
        res = client.post("/api/playground", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["is_video"] is True
        assert data["faces_found"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["best_match"] == "sontung"


