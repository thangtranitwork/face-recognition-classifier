import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from facecls.config import AppConfig
from facecls.io import build_new_name, is_already_processed, extract_candidate_prefix
from facecls.sensitive import (
    check_sensitive,
    check_sensitive_media,
    EXPLICIT_CATEGORIES,
    SENSITIVE_CATEGORIES,
    CATEGORY_LABELS_VI,
)
from facecls.pipeline import run_rename, run_organize, run_clean
from facecls.server import app
from starlette.testclient import TestClient

client = TestClient(app)


def test_build_new_name_sensitive():
    # Normal
    name = build_new_name("sontung", "pic1.jpg", "{person}_{original}", is_sensitive=False)
    assert name == "sontung_pic1.jpg"

    # Sensitive rename
    name_sens = build_new_name("sontung", "pic1.jpg", "{person}_{original}", is_sensitive=True)
    assert name_sens == "sensitive_sontung_pic1.jpg"

    # Re-running on sensitive name doesn't double-prefix
    name_repeat = build_new_name("sontung", "sensitive_sontung_pic1.jpg", "{person}_{original}", is_sensitive=True)
    assert name_repeat == "sensitive_sontung_pic1.jpg"

    # Low confidence format + sensitive
    name_low_sens = build_new_name("sontung", "pic1.jpg", "low_{person}_{original}", is_sensitive=True)
    assert name_low_sens == "sensitive_low_sontung_pic1.jpg"


def test_is_already_processed_and_extract_prefix_with_sensitive():
    known = {"sontung", "huynh_giang"}

    # Sensitive file recognized as already processed
    assert is_already_processed("sensitive_sontung_001.jpg", known, "unknown") is True
    assert is_already_processed("sensitive_low_huynh_giang_002.jpg", known, "unknown") is True

    # Candidate prefix extracted accurately
    assert extract_candidate_prefix("sensitive_sontung_001.jpg", "unknown") == "sontung"
    assert extract_candidate_prefix("sensitive_low_sontung_002.jpg", "unknown") == "sontung"


def test_check_sensitive_mock():
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [
        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.88, "box": [10, 10, 50, 50]},
        {"class": "FACE_FEMALE", "score": 0.95, "box": [5, 5, 20, 20]},
    ]

    with patch("facecls.sensitive.get_nudenet_detector", return_value=mock_detector):
        res = check_sensitive("dummy.jpg", threshold=0.35)
        assert res["is_sensitive"] is True
        assert res["status"] == "UNSAFE"
        assert res["max_category"] == "FEMALE_BREAST_EXPOSED"
        assert res["max_category_vi"] == CATEGORY_LABELS_VI["FEMALE_BREAST_EXPOSED"]
        assert res["max_score"] == 88.0
        assert len(res["detections"]) == 1  # FACE_FEMALE is neutral so not in explicit/sensitive list


def test_run_rename_modes_rename_and_copy(tmp_path):
    # Setup test directory 1 for RENAME
    test_dir_rename = tmp_path / "test_unclass_rename"
    test_dir_rename.mkdir()

    file_rename = test_dir_rename / "raw_girl_rename.jpg"
    file_rename.write_bytes(b"dummy image data")

    cfg = AppConfig()
    cfg.dataset_root = str(tmp_path)
    cfg.unclassified_dir = test_dir_rename.name
    cfg.sensitive_prefix = "sensitive"

    mock_engine = MagicMock()
    mock_engine.engine = "insightface"
    mock_engine.extract_from_file.return_value = [MagicMock()]

    mock_index = MagicMock()
    mock_index.match_single.return_value = ("huynh_giang", "matched", 85.0)

    # 1. Test Mode: RENAME
    sens_rename_mock = {
        "is_sensitive": True,
        "status": "UNSAFE",
        "max_category": "FEMALE_BREAST_EXPOSED",
        "max_category_vi": "Ngực nữ (Lộ)",
        "max_score": 88.0,
        "detections": [],
    }
    with patch("facecls.pipeline.load_known_faces", return_value=({"huynh_giang": []}, mock_index)), \
         patch("facecls.pipeline.check_sensitive", return_value=sens_rename_mock):
        run_rename(cfg=cfg, engine_obj=mock_engine, target_dir=str(test_dir_rename), apply=True, sensitive_mode="rename")

    assert (test_dir_rename / "sensitive_huynh_giang_raw_girl_rename.jpg").exists()

    # 2. Test Mode: COPY
    test_dir_copy = tmp_path / "test_unclass_copy"
    test_dir_copy.mkdir()
    file_copy = test_dir_copy / "raw_girl_copy.jpg"
    file_copy.write_bytes(b"dummy image data")

    cfg.unclassified_dir = test_dir_copy.name

    sens_copy_mock = {
        "is_sensitive": True,
        "status": "SENSITIVE",
        "max_category": "FEMALE_BREAST_COVERED",
        "max_category_vi": "Ngực nữ (Bikini/Đồ lót)",
        "max_score": 75.0,
        "detections": [],
    }
    with patch("facecls.pipeline.load_known_faces", return_value=({"huynh_giang": []}, mock_index)), \
         patch("facecls.pipeline.check_sensitive", return_value=sens_copy_mock):
        run_rename(cfg=cfg, engine_obj=mock_engine, target_dir=str(test_dir_copy), apply=True, sensitive_mode="copy")

    # Original is renamed to normal person name: huynh_giang_raw_girl_copy.jpg
    assert (test_dir_copy / "huynh_giang_raw_girl_copy.jpg").exists()
    # AND a duplicate copy is created with sensitive_ prefix: sensitive_huynh_giang_raw_girl_copy.jpg
    assert (test_dir_copy / "sensitive_huynh_giang_raw_girl_copy.jpg").exists()


def test_api_scan_sensitive_mode_in_stream():
    sens_mock = {
        "is_sensitive": True,
        "status": "UNSAFE",
        "max_category": "FEMALE_GENITALIA_EXPOSED",
        "max_category_vi": "Vùng kín nữ (Lộ)",
        "max_score": 92.0,
        "detections": [],
    }

    with patch("facecls.server.load_known_faces", return_value=({}, None)), \
         patch("facecls.server.check_sensitive", return_value=sens_mock):
        with client.stream("GET", "/api/scan/stream?apply=false&sensitive_mode=rename") as resp:
            assert resp.status_code == 200
            start_event = None
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    import json
                    payload = json.loads(line[5:])
                    if payload["type"] == "start":
                        start_event = payload
                        assert payload["sensitive_mode"] == "rename"
                        break
            assert start_event is not None


def test_check_sensitive_box_pct_and_media():
    import numpy as np
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [
        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.90, "box": [100, 200, 50, 60]}
    ]

    # Test with ndarray (height=400, width=500)
    fake_img = np.zeros((400, 500, 3), dtype=np.uint8)
    with patch("facecls.sensitive.get_nudenet_detector", return_value=mock_detector):
        res = check_sensitive(fake_img, threshold=0.35)
        assert res["is_sensitive"] is True
        assert len(res["detections"]) == 1
        det = res["detections"][0]
        assert det["box_pct"] is not None
        # x: 100/500 = 20%, y: 200/400 = 50%, w: 50/500 = 10%, h: 60/400 = 15%
        assert det["box_pct"]["x"] == 20.0
        assert det["box_pct"]["y"] == 50.0
        assert det["box_pct"]["w"] == 10.0
        assert det["box_pct"]["h"] == 15.0

    # Test check_sensitive_media with frames
    fake_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    with patch("facecls.sensitive.get_nudenet_detector", return_value=mock_detector):
        media_res = check_sensitive_media("dummy.mp4", frames=fake_frames, threshold=0.35)
        assert media_res["is_sensitive"] is True
        assert media_res["is_video"] is True
        assert media_res["max_score"] == 90.0


def test_run_organize_and_clean_with_sensitive(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    person_dir = dataset_root / "sontung"
    person_dir.mkdir()
    unclass_dir = dataset_root / "chua_phan_loai"
    unclass_dir.mkdir()

    sens_file = unclass_dir / "sensitive_sontung_raw1.jpg"
    sens_file.write_bytes(b"dummy")

    cfg = AppConfig()
    cfg.dataset_root = str(dataset_root)
    cfg.unclassified_dir = "chua_phan_loai"
    cfg.sensitive_prefix = "sensitive"

    # Normal file
    norm_file = unclass_dir / "sontung_raw2.jpg"
    norm_file.write_bytes(b"dummy norm")

    # Test run_organize: sensitive -> dataset/sensitive/ (ảnh nằm chung), normal -> dataset/sontung/
    run_organize(cfg=cfg, apply=True, yes=True)

    sens_dest = dataset_root / "sensitive" / "sensitive_sontung_raw1.jpg"
    norm_dest = person_dir / "sontung_raw2.jpg"

    assert sens_dest.exists()
    assert norm_dest.exists()
    assert not sens_file.exists()
    assert not norm_file.exists()

    # Test run_clean preserves sensitive_ prefix
    run_clean(cfg=cfg, target_dir="sontung", apply=True)
    cleaned_norm = person_dir / "sontung_1.jpg"
    assert cleaned_norm.exists()

    # Verify that dataset/sensitive is in exclude_dirs and not treated as a person
    import numpy as np
    from facecls.pipeline import load_known_faces
    mock_engine = MagicMock()
    mock_engine.extract_from_file.return_value = [np.ones(512, dtype=np.float32)]
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    known, _ = load_known_faces(cfg, mock_engine, mock_cache, dataset_root, unclass_dir)
    assert "sensitive" not in known
    assert "sontung" in known


