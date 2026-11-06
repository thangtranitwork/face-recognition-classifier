"""Module cấu hình cho hệ thống nhận diện khuôn mặt."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional
import yaml


@dataclass
class AppConfig:
    # Thư mục dữ liệu
    dataset_root: str = "./dataset"
    unclassified_dir: str = "chua_phan_loai"
    exclude_dirs: List[str] = field(default_factory=lambda: [
        "chua_phan_loai", "khac", "other", "GV", "DD", "N", "Ng",
        "Nhiều", "neko", "S", "ADT", "NCDK"
    ])

    # Nhận diện & mô hình
    engine: str = "insightface"  # "insightface" hoặc "dlib"
    insightface_model: str = "buffalo_l"
    allowed_modules: List[str] = field(default_factory=lambda: ["detection", "recognition"])
    det_thresh: float = 0.6  # Ngưỡng chất lượng khuôn mặt (det_score)
    cosine_threshold: float = 0.35
    multi_face_strategy: str = "largest"  # "largest" hoặc "reject"
    margin_threshold: float = 0.04
    tolerance: float = 0.57  # Cho dlib
    top_k_neighbors: int = 3
    model: str = "hog"  # Cho dlib: "hog" hoặc "cnn"
    num_jitters: int = 2  # Cho dlib
    max_images_per_person: int = 50
    image_max_size: int = 1600  # Kích thước cạnh dài tối đa khi decode ảnh
    random_seed: int = 42  # Seed lấy mẫu ngẫu nhiên để đảm bảo tính nhất quán

    # Xử lý video
    sample_frames_per_video: int = 12

    # Gom cụm (Clustering)
    cluster_eps_cosine: float = 0.40  # Khoảng cách Cosine cho InsightFace
    cluster_eps_dlib: float = 0.38    # Khoảng cách Euclidean cho dlib
    cluster_min_samples: int = 2
    cluster_prefix: str = "nguoi_moi"

    # Đặt tên file & Confidence tiers
    low_confidence_threshold: float = 45.0
    name_format: str = "{person}_{original}"
    low_confidence_format: str = "low_{person}_{original}"
    unknown_prefix: str = "unknown"

    # Bộ nhớ đệm (Cache)
    use_cache: bool = True
    cache_file: str = ".face_encodings_cache.pkl"
    cluster_cache_file: str = ".cluster_encodings_cache.pkl"

    # Logging
    log_to_file: bool = True
    log_file: str = "rename_log.txt"

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Nạp cấu hình từ tệp YAML và áp dụng giá trị mặc định chuẩn."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp cấu hình: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    cfg = AppConfig()

    for k, v in raw_cfg.items():
        if hasattr(cfg, k):
            expected_type = type(getattr(cfg, k))
            if expected_type in (int, float) and v is not None:
                try:
                    setattr(cfg, k, expected_type(v))
                except (ValueError, TypeError):
                    pass
            elif expected_type == bool and v is not None:
                setattr(cfg, k, bool(v))
            else:
                setattr(cfg, k, v)

    # Đảm bảo max_images_per_person là số nguyên
    if cfg.max_images_per_person is not None:
        try:
            cfg.max_images_per_person = int(cfg.max_images_per_person)
        except (ValueError, TypeError):
            cfg.max_images_per_person = 50

    # Tương thích ngược: nếu trong config cũ chỉ có cluster_eps
    if "cluster_eps" in raw_cfg and "cluster_eps_cosine" not in raw_cfg:
        val = float(raw_cfg["cluster_eps"])
        cfg.cluster_eps_cosine = val
        cfg.cluster_eps_dlib = val

    return cfg
