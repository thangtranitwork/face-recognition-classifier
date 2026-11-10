"""Hàm tiện ích nhập/xuất tệp, định dạng ảnh, xử lý tên và hash nội dung."""

import hashlib
import re
import shutil
from pathlib import Path
from typing import Optional, Set
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

VALID_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VALID_VIDEO_EXTENSIONS: Set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALL_VALID_EXTENSIONS: Set[str] = VALID_IMAGE_EXTENSIONS | VALID_VIDEO_EXTENSIONS

INVALID_PREFIXES: Set[str] = {
    "img", "image", "images", "photo", "screenshot", "screencast",
    "unknown", "low", "video", "media", "frame"
}


def resolve_target_dir(target_dir_str: Optional[str], default_name: str, dataset_root: Path) -> Path:
    """Xác định đường dẫn thư mục mục tiêu (tương đối hoặc tuyệt đối)."""
    if target_dir_str:
        p = Path(target_dir_str).expanduser().resolve()
        if p.exists() and p.is_dir():
            return p
        rel_p = dataset_root / target_dir_str
        if rel_p.exists() and rel_p.is_dir():
            return rel_p
        raise FileNotFoundError(f"Thư mục không tồn tại: '{target_dir_str}' (đã tìm ở {p} và {rel_p})")
    return dataset_root / default_name


def compute_file_content_hash(file_path: Path) -> str:
    """
    Tính hash nội dung tệp bằng SHA-256 kết hợp kích thước tệp.
    Đảm bảo định danh duy nhất theo nội dung, không phụ thuộc vào đường dẫn hay tên tệp.
    """
    stat = file_path.stat()
    hasher = hashlib.sha256()
    hasher.update(f"size:{stat.st_size}".encode("utf-8"))

    with open(file_path, "rb") as f:
        # Đọc theo chunk 64KB
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_image_rgb(file_path: Path, max_size: int = 1600) -> Optional[np.ndarray]:
    """
    Đọc ảnh, xử lý xoay chuẩn EXIF, thu nhỏ kích thước nếu quá lớn (để tiết kiệm RAM/CPU)
    và trả về mảng numpy RGB.
    """
    try:
        with Image.open(file_path) as pil_img:
            # Xử lý xoay theo EXIF
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img is None:
                return None
            pil_img = pil_img.convert("RGB")

            # Thu nhỏ nếu cạnh dài vượt quá max_size (tiết kiệm bộ nhớ cho ảnh kích thước lớn)
            w, h = pil_img.size
            max_edge = max(w, h)
            if max_edge > max_size and max_size > 0:
                scale = max_size / float(max_edge)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            return np.array(pil_img)
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    except Exception:
        return None


def generate_unique_path(dest_dir: Path, desired_name: str) -> Path:
    """Tạo đường dẫn tệp không bị trùng lặp bằng cách thêm hậu tố số `_1`, `_2`..."""
    dest_path = dest_dir / desired_name
    if not dest_path.exists():
        return dest_path

    stem = Path(desired_name).stem
    suffix = Path(desired_name).suffix
    counter = 1
    while dest_path.exists():
        new_name = f"{stem}_{counter}{suffix}"
        dest_path = dest_dir / new_name
        counter += 1
    return dest_path


def safe_rename(src_path: Path, new_name: str) -> Path:
    """
    Đổi tên tệp tại chỗ một cách an toàn, tránh ghi đè lên tệp đã tồn tại trên hệ thống Linux.
    """
    dest_path = src_path.parent / new_name
    if dest_path.resolve() == src_path.resolve():
        return src_path

    if dest_path.exists():
        dest_path = generate_unique_path(src_path.parent, new_name)

    src_path.rename(dest_path)
    return dest_path


def safe_move(src_path: Path, target_dir: Path, target_filename: Optional[str] = None) -> Path:
    """
    Di chuyển tệp sang thư mục đích an toàn, tự động đổi tên chống ghi đè dữ liệu.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    fname = target_filename or src_path.name
    dest_path = generate_unique_path(target_dir, fname)
    shutil.move(str(src_path), str(dest_path))
    return dest_path


def build_new_name(
    person: str,
    original_name: str,
    name_format: str,
    is_sensitive: bool = False,
    sensitive_prefix: str = "sensitive"
) -> str:
    """Tạo tên tệp mới theo định dạng khuôn mẫu, tránh trùng lặp tiền tố đã tồn tại."""
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix.lower()

    # Làm sạch stem nếu file đã từng có tiền tố sensitive_, low_, hoặc tên người
    clean_stem = stem
    p_lower = person.lower()
    s_lower = sensitive_prefix.lower()

    changed = True
    while changed:
        changed = False
        if clean_stem.lower().startswith(f"{s_lower}_") or clean_stem.lower().startswith(f"{s_lower}-"):
            clean_stem = clean_stem[len(sensitive_prefix) + 1:]
            changed = True
        if clean_stem.lower().startswith("low_") or clean_stem.lower().startswith("low-"):
            clean_stem = clean_stem[4:]
            changed = True

    if clean_stem.lower().startswith(f"{p_lower}_") or clean_stem.lower().startswith(f"{p_lower}-"):
        clean_stem = clean_stem[len(person) + 1:]

    new_stem = name_format.format(person=person, original=clean_stem)
    if is_sensitive:
        new_stem = f"{sensitive_prefix}_{new_stem}"
    return f"{new_stem}{suffix}"


def is_already_processed(filename: str, known_person_names: Set[str], unknown_prefix: str) -> bool:
    """Kiểm tra xem tệp đã từng được xử lý hoặc gán nhãn trước đó hay chưa."""
    name_lower = filename.lower()
    if name_lower.startswith("sensitive_") or name_lower.startswith("sensitive-"):
        name_lower = name_lower[10:]
    if name_lower.startswith(f"{unknown_prefix.lower()}_") or name_lower.startswith(f"{unknown_prefix.lower()}-"):
        return True
    if name_lower.startswith("low_") or name_lower.startswith("low-"):
        return True
    for person in known_person_names:
        p_lower = person.lower()
        if name_lower.startswith(f"{p_lower}_") or name_lower.startswith(f"{p_lower}-"):
            return True
    return False


def extract_candidate_prefix(filename: str, unknown_prefix: str) -> Optional[str]:
    """Trích xuất tiền tố định danh người từ tên tệp."""
    name_lower = filename.lower()
    if name_lower.startswith("sensitive_") or name_lower.startswith("sensitive-"):
        name_lower = name_lower[10:]
    if name_lower.startswith("low_") or name_lower.startswith("low-"):
        name_lower = name_lower[4:]

    if name_lower.startswith(f"{unknown_prefix.lower()}_") or name_lower.startswith(f"{unknown_prefix.lower()}-"):
        return None

    # Khớp tiền tố nhóm người mới ví dụ: nguoi_moi_1, nguoi-moi-1...
    m = re.match(r"^(nguoi[_\-]moi[_\-]\d+)[_\-]", name_lower)
    if m:
        return m.group(1).replace("-", "_")

    # Tách tiền tố trước dấu _ hoặc - đầu tiên
    parts = re.split(r"[_\-]", name_lower)
    if len(parts) >= 2:
        if parts[0] == "nguoi" and len(parts) >= 3 and parts[1] == "moi":
            return f"{parts[0]}_{parts[1]}_{parts[2]}"
        candidate = parts[0].strip().lower()
        if candidate not in INVALID_PREFIXES and not candidate.isdigit() and len(candidate) > 1:
            return candidate

    return None
