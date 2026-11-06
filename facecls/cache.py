"""Hệ thống bộ nhớ đệm (Cache) embedding thông minh theo hash nội dung tệp."""

import os
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from facecls.io import compute_file_content_hash

logger = logging.getLogger("face_renamer")


class EmbeddingCache:
    """
    Quản lý cache embedding khuôn mặt.
    Sử dụng 2 tầng lưu trữ:
    1. path_index: {abs_path: (mtime, size, content_hash)} -> Tra cứu siêu nhanh không cần đọc lại tệp.
    2. embedding_store: {f"{engine}:{model}:{content_hash}": np.ndarray} -> Đảm bảo khi đổi tên hoặc di chuyển tệp
       vẫn trúng cache 100%, dùng chung hoàn hảo giữa rename_faces và cluster_faces.
    """

    def __init__(self, cache_file_path: Path):
        self.cache_file_path = cache_file_path
        self.path_index: Dict[str, tuple[float, int, str]] = {}
        self.embedding_store: Dict[str, np.ndarray] = {}
        self._dirty = False
        self.load()

    def _get_hash(self, file_path: Path) -> str:
        abs_p = str(file_path.resolve())
        stat = file_path.stat()
        mtime, size = stat.st_mtime, stat.st_size

        if abs_p in self.path_index:
            cached_mtime, cached_size, cached_hash = self.path_index[abs_p]
            if cached_mtime == mtime and cached_size == size:
                return cached_hash

        # Nếu tệp mới hoặc đã sửa đổi -> tính lại hash
        h = compute_file_content_hash(file_path)
        self.path_index[abs_p] = (mtime, size, h)
        self._dirty = True
        return h

    def get(self, file_path: Path, engine: str, model_name: str) -> Optional[np.ndarray]:
        """Lấy embedding từ cache nếu có."""
        try:
            if not file_path.exists() or not file_path.is_file():
                return None
            content_hash = self._get_hash(file_path)
            key = f"{engine}:{model_name}:{content_hash}"
            return self.embedding_store.get(key)
        except Exception:
            return None

    def set(self, file_path: Path, engine: str, model_name: str, embedding: Optional[np.ndarray]) -> None:
        """Lưu embedding vào cache (hoặc đánh dấu không có mặt bằng mảng rỗng/None)."""
        try:
            if not file_path.exists():
                return
            content_hash = self._get_hash(file_path)
            key = f"{engine}:{model_name}:{content_hash}"
            if embedding is not None:
                self.embedding_store[key] = np.asarray(embedding, dtype=np.float32)
            else:
                self.embedding_store[key] = np.empty(0, dtype=np.float32)
            self._dirty = True
        except Exception:
            pass

    def load(self) -> None:
        """Đọc cache từ đĩa."""
        if not self.cache_file_path.exists():
            return
        try:
            with open(self.cache_file_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                # Hỗ trợ cấu trúc cache mới
                if "embedding_store" in data and "path_index" in data:
                    self.embedding_store = data.get("embedding_store", {})
                    self.path_index = data.get("path_index", {})
                # Di chuyển từ định dạng cũ nếu có
                elif "encodings" in data:
                    raw = data["encodings"]
                    for path_str, item in raw.items():
                        enc = item.get("encoding")
                        if enc is not None:
                            p = Path(path_str)
                            if p.exists():
                                try:
                                    h = compute_file_content_hash(p)
                                    stat = p.stat()
                                    self.path_index[str(p.resolve())] = (stat.st_mtime, stat.st_size, h)
                                    # Default key
                                    self.embedding_store[f"insightface:buffalo_l:{h}"] = np.asarray(enc, dtype=np.float32)
                                except Exception:
                                    pass
        except Exception as e:
            logger.warning(f"⚠️ Không đọc được tệp cache ({e}), khởi tạo cache mới.")
            self.embedding_store = {}
            self.path_index = {}

    def save(self) -> None:
        """Ghi cache ra đĩa nếu có thay đổi."""
        if not self._dirty:
            return
        try:
            self.cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file_path, "wb") as f:
                pickle.dump({
                    "version": 2,
                    "embedding_store": self.embedding_store,
                    "path_index": self.path_index,
                }, f)
            self._dirty = False
        except Exception as e:
            logger.warning(f"⚠️ Không thể lưu tệp cache: {e}")

    def clear(self) -> None:
        """Xóa toàn bộ cache."""
        self.embedding_store.clear()
        self.path_index.clear()
        self._dirty = False
        if self.cache_file_path.exists():
            try:
                self.cache_file_path.unlink()
            except Exception:
                pass
