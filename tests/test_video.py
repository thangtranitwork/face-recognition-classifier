import pytest
from pathlib import Path
import tempfile
import cv2
import numpy as np
from facecls.video import is_video_url, sample_video_frames, analyze_video_motion_and_best_frame

def test_is_video_url():
    assert is_video_url("https://www.facebook.com/stories/highlights/100050246662860/")
    assert is_video_url("https://www.facebook.com/reel/1234567890")
    assert is_video_url("https://www.facebook.com/user/videos/123456")
    assert is_video_url("https://www.instagram.com/stories/highlights/18090946048123978/")
    assert is_video_url("https://www.instagram.com/reel/C12345678/")
    assert is_video_url("https://example.com/sample.mp4")
    assert is_video_url("https://example.com/sample.mov")
    assert not is_video_url("https://www.facebook.com/caca.tieu.50309/photos")
    assert not is_video_url("Sơn Tùng M-TP")

def test_sample_video_frames():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, 10, (64, 64))
        for i in range(20):
            frame = np.full((64, 64, 3), i * 10, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        frames = list(sample_video_frames(tmp_path, sample_interval_sec=0.5, max_frames=5))
        assert len(frames) > 0
        idx, t_sec, rgb = frames[0]
        assert idx == 0
        assert rgb.shape == (64, 64, 3)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_analyze_video_motion_static():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, 10, (120, 120))
        base_frame = np.full((120, 120, 3), 150, dtype=np.uint8)
        cv2.circle(base_frame, (60, 60), 30, (50, 100, 200), -1)
        for _ in range(25):
            writer.write(base_frame)
        writer.release()

        is_static, best_frame, reason = analyze_video_motion_and_best_frame(tmp_path)
        assert is_static is True
        assert best_frame is not None
        assert best_frame.shape == (120, 120, 3)
        assert "tĩnh" in reason.lower() or "static" in reason.lower()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_analyze_video_motion_moving():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, 10, (120, 120))
        for i in range(25):
            frame = np.full((120, 120, 3), 100, dtype=np.uint8)
            cv2.circle(frame, (20 + i * 3, 60), 20, (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        is_static, best_frame, reason = analyze_video_motion_and_best_frame(tmp_path)
        assert is_static is False
        assert best_frame is not None
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

