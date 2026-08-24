#!/usr/bin/env python3
"""
Face Recognition File Renamer
==============================
Tự động nhận diện khuôn mặt trong ảnh chưa phân loại và đổi tên file
theo người được nhận diện.

Cách dùng:
  # Chạy dry-run (chỉ xem kết quả, không đổi tên)
  python rename_faces.py

  # Thực sự đổi tên
  python rename_faces.py --apply

  # Chỉ định file config khác
  python rename_faces.py --config /path/to/config.yaml

  # Xóa cache và build lại
  python rename_faces.py --clear-cache

  # Xem thêm options
  python rename_faces.py --help
"""

import os
import sys
import pickle
import hashlib
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import yaml
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

try:
    import face_recognition
except ImportError:
    print("❌ Thiếu thư viện face_recognition. Chạy: pip install face_recognition")
    sys.exit(1)


# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────

def setup_logger(log_to_file: bool, log_file: str, dataset_root: Path) -> logging.Logger:
    logger = logging.getLogger("face_renamer")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_to_file:
        log_path = dataset_root / log_file
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load và validate config từ file YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Defaults
    cfg.setdefault("tolerance", 0.55)
    cfg.setdefault("model", "hog")
    cfg.setdefault("max_images_per_person", 50)
    cfg.setdefault("unknown_prefix", "unknown")
    cfg.setdefault("name_format", "{person}_{original}")
    cfg.setdefault("low_confidence_format", "low_{person}_{original}")
    cfg.setdefault("low_confidence_threshold", 70)
    cfg.setdefault("use_cache", True)
    cfg.setdefault("cache_file", ".face_encodings_cache.pkl")
    cfg.setdefault("log_to_file", True)
    cfg.setdefault("log_file", "rename_log.txt")
    cfg.setdefault("exclude_dirs", [])

    return cfg


# ─────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────

def _compute_dir_fingerprint(person_dirs: list[Path]) -> str:
    """
    Tạo fingerprint dựa trên số lượng file + mtime của folder.
    Tối ưu cho dataset lớn (60 người × vài trăm ảnh):
    chỉ stat folder thay vì scan từng file.
    """
    hasher = hashlib.md5()
    for d in sorted(person_dirs):
        dir_stat = d.stat()
        # Đếm số file ảnh để phát hiện thêm/xóa
        file_count = sum(1 for p in d.iterdir() if p.is_file())
        hasher.update(f"{d.name}:{dir_stat.st_mtime}:{file_count}".encode())
    return hasher.hexdigest()


def load_cache(cache_path: Path, fingerprint: str, logger: logging.Logger) -> Optional[dict]:
    """Load cache nếu còn hợp lệ (fingerprint khớp)."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if data.get("fingerprint") != fingerprint:
            logger.info("🔄 Cache cũ (dataset thay đổi), sẽ build lại...")
            return None
        logger.info(f"✅ Dùng cache từ {cache_path}")
        return data["known_faces"]
    except Exception as e:
        logger.warning(f"⚠️  Không đọc được cache: {e}. Build lại...")
        return None


def save_cache(cache_path: Path, known_faces: dict, fingerprint: str, logger: logging.Logger):
    """Lưu cache ra file."""
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "known_faces": known_faces}, f)
        logger.info(f"💾 Đã lưu cache → {cache_path}")
    except Exception as e:
        logger.warning(f"⚠️  Không lưu được cache: {e}")


# ─────────────────────────────────────────────
# Known faces loader
# ─────────────────────────────────────────────

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALL_VALID_EXTENSIONS = VALID_EXTENSIONS | VALID_VIDEO_EXTENSIONS


def _extract_frames_from_video(video_path: Path, num_frames: int = 12) -> list[np.ndarray]:
    """Trích xuất num_frames khung hình trải đều từ video."""
    try:
        import cv2
    except ImportError:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    step = max(1, total_frames // num_frames)
    frame_indices = [i * step for i in range(min(num_frames, total_frames))]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

    cap.release()
    return frames


def recognize_video(
    video_path: Path,
    known_faces: dict[str, list],
    tolerance: float,
    model: str,
    unknown_prefix: str,
    sample_frames: int,
    logger: logging.Logger,
) -> tuple[str, str, float]:
    """
    Nhận diện khuôn mặt trong video bằng cách trích xuất khung hình và bầu chọn (Majority Voting).
    """
    frames = _extract_frames_from_video(video_path, num_frames=sample_frames)
    if not frames:
        return unknown_prefix, "không đọc được video / video trống", 0.0

    person_votes: dict[str, list[float]] = {}
    valid_frames_count = 0

    for frame in frames:
        locations = face_recognition.face_locations(frame, model=model)
        if not locations:
            continue
        encodings = face_recognition.face_encodings(frame, locations)
        if not encodings:
            continue

        valid_frames_count += 1
        for face_enc in encodings:
            best_person = None
            best_distance = float("inf")
            for person_name, known_encs in known_faces.items():
                distances = face_recognition.face_distance(known_encs, face_enc)
                min_dist = float(np.min(distances))
                if min_dist < best_distance:
                    best_distance = min_dist
                    best_person = person_name

            if best_distance <= tolerance:
                conf = round((1 - best_distance) * 100, 1)
                person_votes.setdefault(best_person, []).append(conf)

    if not person_votes:
        return unknown_prefix, f"video ({len(frames)} frames): không nhận diện được ai", 0.0

    top_person, confs = max(person_votes.items(), key=lambda item: (len(item[1]), np.mean(item[1])))
    avg_conf = round(float(np.mean(confs)), 1)
    votes_count = len(confs)

    reason = f"video: {votes_count}/{valid_frames_count} frames khớp {top_person} (conf trung bình {avg_conf}%)"
    return top_person, reason, avg_conf



def load_known_faces(
    dataset_root: Path,
    exclude_dirs: list[str],
    max_images_per_person: Optional[int],
    model: str,
    cache_path: Path,
    use_cache: bool,
    logger: logging.Logger,
) -> dict[str, list]:
    """
    Quét tất cả folder con (trừ exclude_dirs), encode khuôn mặt,
    trả về dict {person_name: [encoding, ...]}.
    """
    # Thu thập các folder person
    exclude_set = {d.lower() for d in exclude_dirs}
    person_dirs = [
        d for d in dataset_root.iterdir()
        if d.is_dir() and d.name.lower() not in exclude_set
    ]

    if not person_dirs:
        logger.error("❌ Không tìm thấy folder person nào trong dataset_root.")
        sys.exit(1)

    logger.info(f"📂 Tìm thấy {len(person_dirs)} folder người: {[d.name for d in person_dirs]}")

    # Kiểm tra cache
    if use_cache:
        fingerprint = _compute_dir_fingerprint(person_dirs)
        cached = load_cache(cache_path, fingerprint, logger)
        if cached is not None:
            return cached

    # Build encodings
    known_faces: dict[str, list] = {}

    for person_dir in person_dirs:
        person_name = person_dir.name
        all_images = [
            p for p in person_dir.rglob("*")
            if p.suffix.lower() in VALID_EXTENSIONS
        ]

        if not all_images:
            logger.warning(f"⚠️  Folder '{person_name}' không có ảnh nào, bỏ qua.")
            continue

        # Shuffle để đảm bảo đa dạng khi dừng sớm
        shuffled = all_images.copy()
        random.shuffle(shuffled)

        encodings = []
        no_face_count = 0
        processed_count = 0
        target = max_images_per_person  # None = lấy hết

        for img_path in tqdm(shuffled, desc=f"  Encoding {person_name}", leave=False, unit="img"):
            # Đạt đủ encodings mục tiêu thì dừng
            if target and len(encodings) >= target:
                break

            enc = _encode_image(img_path, model, logger)
            processed_count += 1

            if enc:
                encodings.extend(enc)
            else:
                no_face_count += 1

        # Log chi tiết per-person
        face_images = processed_count - no_face_count
        skipped = len(all_images) - processed_count
        if no_face_count > 0:
            logger.info(
                f"  👤 {person_name}: {len(all_images)} ảnh tổng | "
                f"{face_images} có mặt ✅ | {no_face_count} không có mặt ⚠️ | "
                f"bỏ qua {skipped} | → {len(encodings)} encodings"
            )
        else:
            logger.info(
                f"  👤 {person_name}: {processed_count}/{len(all_images)} ảnh → {len(encodings)} encodings"
            )

        if encodings:
            known_faces[person_name] = encodings
        else:
            logger.warning(f"⚠️  '{person_name}': không tìm thấy khuôn mặt nào trong {processed_count} ảnh đã thử!")


    if not known_faces:
        logger.error("❌ Không có known face nào. Kiểm tra lại dataset.")
        sys.exit(1)

    # Lưu cache
    if use_cache:
        save_cache(cache_path, known_faces, fingerprint, logger)

    return known_faces


def _encode_image(img_path: Path, model: str, logger: logging.Logger) -> list:
    """Load ảnh và trả về list encodings (1 ảnh có thể có nhiều khuôn mặt)."""
    try:
        # Dùng Pillow để đảm bảo đọc được nhiều format
        pil_img = Image.open(img_path).convert("RGB")
        img_array = np.array(pil_img)
        encs = face_recognition.face_encodings(
            img_array,
            face_recognition.face_locations(img_array, model=model),
        )
        return list(encs)
    except UnidentifiedImageError:
        logger.debug(f"     ⚠️  Không đọc được ảnh: {img_path.name}")
        return []
    except Exception as e:
        logger.debug(f"     ⚠️  Lỗi encode {img_path.name}: {e}")
        return []


# ─────────────────────────────────────────────
# Recognition
# ─────────────────────────────────────────────

def recognize_person(
    img_path: Path,
    known_faces: dict[str, list],
    tolerance: float,
    model: str,
    unknown_prefix: str,
    logger: logging.Logger,
) -> tuple[str, str, float]:
    """
    Nhận diện khuôn mặt trong ảnh.

    Returns:
        (person_name, reason, confidence)
        - person_name: tên người hoặc unknown_prefix
        - reason: mô tả lý do (để log)
        - confidence: 0.0-100.0 (0.0 nếu unknown)
    """
    try:
        pil_img = Image.open(img_path).convert("RGB")
        img_array = np.array(pil_img)
    except (UnidentifiedImageError, Exception) as e:
        return unknown_prefix, f"không đọc được ảnh ({e})", 0.0

    locations = face_recognition.face_locations(img_array, model=model)

    # Không có khuôn mặt nào
    if len(locations) == 0:
        return unknown_prefix, "không phát hiện khuôn mặt", 0.0

    # Nhiều hơn 1 khuôn mặt
    if len(locations) > 1:
        return unknown_prefix, f"có {len(locations)} khuôn mặt trong ảnh", 0.0

    # Đúng 1 khuôn mặt → thử nhận diện
    encodings = face_recognition.face_encodings(img_array, locations)
    if not encodings:
        return unknown_prefix, "không encode được khuôn mặt", 0.0

    face_enc = encodings[0]
    best_person = None
    best_distance = float("inf")

    for person_name, known_encs in known_faces.items():
        distances = face_recognition.face_distance(known_encs, face_enc)
        min_dist = float(np.min(distances))
        if min_dist < best_distance:
            best_distance = min_dist
            best_person = person_name

    if best_distance <= tolerance:
        confidence = round((1 - best_distance) * 100, 1)
        return best_person, f"confidence {confidence}% (distance={best_distance:.3f})", confidence
    else:
        return unknown_prefix, f"không khớp (min_distance={best_distance:.3f} > tolerance={tolerance})", 0.0


# ─────────────────────────────────────────────
# Renaming logic
# ─────────────────────────────────────────────

def build_new_name(person: str, original_name: str, name_format: str, unknown_prefix: str) -> str:
    """Tạo tên file mới theo format cấu hình."""
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix.lower()
    new_stem = name_format.format(person=person, original=stem)
    return f"{new_stem}{suffix}"


def is_already_processed(filename: str, known_faces: dict, unknown_prefix: str) -> bool:
    """Kiểm tra xem file đã được đổi tên trước đó chưa."""
    name_lower = filename.lower()
    # Các prefix hệ thống
    if name_lower.startswith(unknown_prefix.lower() + "_"):
        return True
    if name_lower.startswith("low_"):
        return True
    for person in known_faces:
        if name_lower.startswith(person.lower() + "_"):
            return True
    return False


def process_unclassified(
    unclassified_dir: Path,
    known_faces: dict,
    cfg: dict,
    apply: bool,
    logger: logging.Logger,
) -> dict:
    """
    Quét và đổi tên (hoặc dry-run) ảnh trong thư mục chưa phân loại.

    Returns:
        stats dict với số lượng từng loại kết quả.
    """
    media_files = [
        p for p in unclassified_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ALL_VALID_EXTENSIONS
    ]

    if not media_files:
        logger.warning(f"⚠️  Không có ảnh/video nào trong {unclassified_dir}")
        return {}

    logger.info(f"\n📁 Xử lý {len(media_files)} file (ảnh/video) trong '{unclassified_dir.name}/'")
    if not apply:
        logger.info("🔍 CHẾ ĐỘ DRY-RUN (chỉ xem kết quả, dùng --apply để thực sự đổi tên)\n")

    low_threshold = cfg.get("low_confidence_threshold", 70)
    sample_frames = cfg.get("sample_frames_per_video", 12)
    stats = {"recognized": 0, "low_confidence": 0, "unknown": 0, "skipped": 0, "error": 0}
    results = []

    for file_path in tqdm(media_files, desc="Đang nhận diện", unit="file"):
        original_name = file_path.name

        # Bỏ qua file đã xử lý
        if is_already_processed(original_name, known_faces, cfg["unknown_prefix"]):
            logger.debug(f"  ⏭️  Bỏ qua (đã xử lý): {original_name}")
            stats["skipped"] += 1
            continue

        if file_path.suffix.lower() in VALID_VIDEO_EXTENSIONS:
            person, reason, confidence = recognize_video(
                file_path,
                known_faces,
                cfg["tolerance"],
                cfg["model"],
                cfg["unknown_prefix"],
                sample_frames,
                logger,
            )
        else:
            person, reason, confidence = recognize_person(
                file_path,
                known_faces,
                cfg["tolerance"],
                cfg["model"],
                cfg["unknown_prefix"],
                logger,
            )

        is_unknown = person == cfg["unknown_prefix"]

        # Chọn format tên dựa trên confidence tier
        if is_unknown:
            name_fmt = cfg["name_format"]  # sẽ không dùng (unknown_prefix thay thế)
        elif confidence < low_threshold:
            name_fmt = cfg.get("low_confidence_format", "low_{person}_{original}")
        else:
            name_fmt = cfg["name_format"]

        new_name = build_new_name(person, original_name, name_fmt, cfg["unknown_prefix"])
        new_path = file_path.parent / new_name

        # Tránh ghi đè file đã tồn tại
        if new_path.exists() and new_path != file_path:
            counter = 1
            stem = Path(new_name).stem
            suffix = Path(new_name).suffix
            while new_path.exists():
                new_name = f"{stem}_{counter}{suffix}"
                new_path = file_path.parent / new_name
                counter += 1

        results.append({
            "original": original_name,
            "new_name": new_name,
            "person": person,
            "reason": reason,
            "confidence": confidence,
            "is_unknown": is_unknown,
        })

        if is_unknown:
            stats["unknown"] += 1
            logger.info(f"  ❓ {original_name} → {new_name}  [{reason}]")
        elif confidence < low_threshold:
            stats["low_confidence"] += 1
            logger.info(f"  ⚠️  {original_name} → {new_name}  [{reason}]")
        else:
            stats["recognized"] += 1
            logger.info(f"  ✅ {original_name} → {new_name}  [{reason}]")

        # Thực sự đổi tên
        if apply:
            try:
                file_path.rename(new_path)
            except Exception as e:
                logger.error(f"  ❌ Lỗi đổi tên {original_name}: {e}")
                stats["error"] += 1
                if is_unknown:
                    stats["unknown"] -= 1
                elif confidence < low_threshold:
                    stats["low_confidence"] -= 1
                else:
                    stats["recognized"] -= 1

    return stats


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

@click.command()
@click.option(
    "--config", "-c",
    default="config.yaml",
    show_default=True,
    help="Đường dẫn tới file cấu hình YAML.",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Thực sự đổi tên file (mặc định là dry-run).",
)
@click.option(
    "--clear-cache",
    is_flag=True,
    default=False,
    help="Xóa cache encodings và build lại từ đầu.",
)
@click.option(
    "--unclassified-dir", "-u",
    default=None,
    help="Override tên thư mục chưa phân loại (ghi đè config).",
)
def main(config: str, apply: bool, clear_cache: bool, unclassified_dir: Optional[str]):
    """
    \b
    Face Recognition File Renamer
    ─────────────────────────────
    Nhận diện khuôn mặt và đổi tên ảnh trong thư mục chưa phân loại.

    \b
    Ví dụ:
      python rename_faces.py                    # dry-run
      python rename_faces.py --apply            # thực sự đổi tên
      python rename_faces.py --clear-cache      # xóa cache
    """
    # ── Load config ──────────────────────────────
    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)

    dataset_root = Path(cfg["dataset_root"]).expanduser().resolve()
    if not dataset_root.exists():
        click.echo(f"❌ dataset_root không tồn tại: {dataset_root}", err=True)
        sys.exit(1)

    # Override unclassified_dir nếu được truyền qua CLI
    unclassified_name = unclassified_dir or cfg["unclassified_dir"]
    unclassified_path = dataset_root / unclassified_name

    if not unclassified_path.exists():
        click.echo(f"❌ Thư mục chưa phân loại không tồn tại: {unclassified_path}", err=True)
        sys.exit(1)

    # Đảm bảo unclassified_dir luôn nằm trong exclude_dirs
    exclude_dirs: list[str] = cfg["exclude_dirs"]
    if unclassified_name not in exclude_dirs:
        exclude_dirs.append(unclassified_name)

    cache_path = dataset_root / cfg["cache_file"]

    # ── Logger ───────────────────────────────────
    logger = setup_logger(cfg["log_to_file"], cfg["log_file"], dataset_root)

    logger.info("=" * 60)
    logger.info("  Face Recognition File Renamer")
    logger.info(f"  Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info(f"📌 Dataset root    : {dataset_root}")
    logger.info(f"📌 Unclassified    : {unclassified_path}")
    logger.info(f"📌 Exclude dirs    : {exclude_dirs}")
    logger.info(f"📌 Tolerance       : {cfg['tolerance']}")
    logger.info(f"📌 Model           : {cfg['model']}")
    logger.info(f"📌 Max img/person  : {cfg['max_images_per_person']}")
    logger.info(f"📌 Cache           : {'on' if cfg['use_cache'] else 'off'}")
    logger.info(f"📌 Low conf (<%)   : {cfg.get('low_confidence_threshold', 70)}%")
    logger.info(f"📌 Mode            : {'APPLY ✏️' if apply else 'DRY-RUN 🔍'}")
    logger.info("")

    # ── Xóa cache nếu được yêu cầu ───────────────
    if clear_cache and cache_path.exists():
        cache_path.unlink()
        logger.info(f"🗑️  Đã xóa cache: {cache_path}")

    # ── Load known faces ─────────────────────────
    logger.info("📚 Loading known faces...")
    known_faces = load_known_faces(
        dataset_root=dataset_root,
        exclude_dirs=exclude_dirs,
        max_images_per_person=cfg["max_images_per_person"],
        model=cfg["model"],
        cache_path=cache_path,
        use_cache=cfg["use_cache"],
        logger=logger,
    )
    logger.info(f"\n✅ Loaded {len(known_faces)} người: {list(known_faces.keys())}")
    total_encs = sum(len(v) for v in known_faces.values())
    logger.info(f"   Tổng {total_encs} face encodings\n")

    # ── Xử lý ảnh chưa phân loại ─────────────────
    stats = process_unclassified(
        unclassified_dir=unclassified_path,
        known_faces=known_faces,
        cfg=cfg,
        apply=apply,
        logger=logger,
    )

    # ── Tổng kết ─────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  KẾT QUẢ")
    logger.info("=" * 60)
    logger.info(f"  ✅ Nhận diện được : {stats.get('recognized', 0)}")
    logger.info(f"  ⚠️  Low confidence : {stats.get('low_confidence', 0)}  (< {cfg.get('low_confidence_threshold', 70)}%)")
    logger.info(f"  ❓ Unknown        : {stats.get('unknown', 0)}")
    logger.info(f"  ⏭️  Đã xử lý trước : {stats.get('skipped', 0)}")
    logger.info(f"  ❌ Lỗi            : {stats.get('error', 0)}")
    if not apply:
        logger.info("\n  💡 Chạy với --apply để thực sự đổi tên.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
