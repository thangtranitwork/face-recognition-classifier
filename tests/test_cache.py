"""Kiểm thử đơn vị cho module cache."""

from pathlib import Path
import numpy as np
import pytest

from facecls.cache import EmbeddingCache


def test_embedding_cache_set_and_get(tmp_path: Path):
    cache_file = tmp_path / "test_cache.pkl"
    cache = EmbeddingCache(cache_file)

    dummy_img = tmp_path / "img1.jpg"
    dummy_img.write_bytes(b"image data 1")

    vec = np.ones(512, dtype=np.float32)
    cache.set(dummy_img, "insightface", "buffalo_l", vec)

    # Lấy lại từ memory
    cached = cache.get(dummy_img, "insightface", "buffalo_l")
    assert cached is not None
    assert np.allclose(cached, vec)

    # Lưu và nạp lại từ disk
    cache.save()
    cache2 = EmbeddingCache(cache_file)
    cached2 = cache2.get(dummy_img, "insightface", "buffalo_l")
    assert cached2 is not None
    assert np.allclose(cached2, vec)


def test_cache_hit_after_rename(tmp_path: Path):
    cache_file = tmp_path / "test_cache.pkl"
    cache = EmbeddingCache(cache_file)

    img = tmp_path / "before.jpg"
    img.write_bytes(b"unique image bytes 98765")

    vec = np.full(512, 0.5, dtype=np.float32)
    cache.set(img, "insightface", "buffalo_l", vec)
    cache.save()

    # Đổi tên file sang after.jpg
    renamed = tmp_path / "after.jpg"
    img.rename(renamed)

    # Nạp cache và kiểm tra file mới đổi tên vẫn trúng cache 100% nhờ content hash
    cache_reloaded = EmbeddingCache(cache_file)
    cached = cache_reloaded.get(renamed, "insightface", "buffalo_l")
    assert cached is not None
    assert np.allclose(cached, vec)
