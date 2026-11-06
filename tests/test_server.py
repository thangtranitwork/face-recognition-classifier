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

def test_api_scan_dry_run():
    response = client.post("/api/scan", json={"apply": False})
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert "results" in data
    assert data["applied"] is False
    if data["results"]:
        first = data["results"][0]
        assert "current_name" in first
        assert "url" in first

from unittest.mock import patch

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

