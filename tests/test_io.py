"""Kiểm thử đơn vị cho module io."""

from pathlib import Path
import pytest
import numpy as np

from facecls.io import (
    generate_unique_path,
    safe_rename,
    safe_move,
    build_new_name,
    is_already_processed,
    extract_candidate_prefix,
    compute_file_content_hash,
)


def test_generate_unique_path(tmp_path: Path):
    f1 = tmp_path / "photo.jpg"
    f1.write_text("dummy")

    p = generate_unique_path(tmp_path, "photo.jpg")
    assert p.name == "photo_1.jpg"

    (tmp_path / "photo_1.jpg").write_text("dummy1")
    p2 = generate_unique_path(tmp_path, "photo.jpg")
    assert p2.name == "photo_2.jpg"


def test_safe_rename_avoid_overwrite(tmp_path: Path):
    src = tmp_path / "img1.jpg"
    src.write_text("source")

    existing = tmp_path / "target.jpg"
    existing.write_text("existing_target")

    # Rename src to target.jpg -> must not overwrite existing
    final_path = safe_rename(src, "target.jpg")
    assert final_path.exists()
    assert final_path.name == "target_1.jpg"
    assert existing.read_text() == "existing_target"
    assert final_path.read_text() == "source"


def test_safe_move(tmp_path: Path):
    src = tmp_path / "file.txt"
    src.write_text("move me")

    dest_dir = tmp_path / "sub"
    dest_dir.mkdir()
    (dest_dir / "file.txt").write_text("already here")

    moved = safe_move(src, dest_dir)
    assert moved.parent == dest_dir
    assert moved.name == "file_1.txt"
    assert moved.read_text() == "move me"
    assert (dest_dir / "file.txt").read_text() == "already here"


def test_build_new_name():
    res = build_new_name("sontung", "IMG_001.JPG", "{person}_{original}")
    assert res == "sontung_IMG_001.jpg"

    # Không nhân đôi tiền tố nếu file đã có tên người đó
    res_no_dup = build_new_name("binhan", "binhan_IMG_0001.webp", "{person}_{original}")
    assert res_no_dup == "binhan_IMG_0001.webp"

    # Chuyển đổi từ low_ sang tên chuẩn mà không trùng lặp
    res_clean_low = build_new_name("binhan", "low_binhan_IMG_0001.webp", "{person}_{original}")
    assert res_clean_low == "binhan_IMG_0001.webp"


def test_is_already_processed():
    known = {"sontung", "binhan"}
    assert is_already_processed("sontung_01.jpg", known, "unknown") is True
    assert is_already_processed("low_binhan_02.png", known, "unknown") is True
    assert is_already_processed("unknown_99.jpg", known, "unknown") is True
    assert is_already_processed("IMG_1234.jpg", known, "unknown") is False


def test_extract_candidate_prefix():
    assert extract_candidate_prefix("nguoi_moi_1_IMG_001.jpg", "unknown") == "nguoi_moi_1"
    assert extract_candidate_prefix("low_nguoi_moi_2_01.jpg", "unknown") == "nguoi_moi_2"
    assert extract_candidate_prefix("binhan_123.jpg", "unknown") == "binhan"
    assert extract_candidate_prefix("unknown_01.jpg", "unknown") is None
    assert extract_candidate_prefix("IMG_123.jpg", "unknown") is None


def test_compute_file_content_hash(tmp_path: Path):
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"hello face recognition 12345")
    f2.write_bytes(b"hello face recognition 12345")

    h1 = compute_file_content_hash(f1)
    h2 = compute_file_content_hash(f2)
    assert h1 == h2
    assert len(h1) == 64
