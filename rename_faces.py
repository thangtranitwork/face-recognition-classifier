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
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

try:
    import face_recognition
except ImportError:
    pass

_INSIGHTFACE_APP = None


def get_insightface_app(model_name: str = "buffalo_sc") -> Optional[object]:
    """Khởi tạo hoặc lấy lại ứng dụng InsightFace (RetinaFace + ArcFace)."""
    global _INSIGHTFACE_APP
    if _INSIGHTFACE_APP is not None:
        return _INSIGHTFACE_APP

    try:
        import warnings
        import insightface
        from insightface.app import FaceAnalysis
        import logging as _logging
        _logging.getLogger('insightface').setLevel(_logging.WARNING)
        # Sửa FutureWarning từ skimage.transform
        warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')

        app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _INSIGHTFACE_APP = app
        return _INSIGHTFACE_APP
    except Exception as e:
        print(f"⚠️  Không khởi tạo được InsightFace: {e}")
        return None


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
    cfg.setdefault("engine", "insightface")
    cfg.setdefault("insightface_model", "buffalo_l")
    cfg.setdefault("cosine_threshold", 0.35)
    cfg.setdefault("multi_face_strategy", "largest")
    cfg.setdefault("tolerance", 0.57)
    cfg.setdefault("top_k_neighbors", 3)
    cfg.setdefault("margin_threshold", 0.04)
    cfg.setdefault("num_jitters", 2)
    cfg.setdefault("model", "hog")
    cfg.setdefault("max_images_per_person", 50)
    cfg.setdefault("unknown_prefix", "unknown")
    cfg.setdefault("name_format", "{person}_{original}")
    cfg.setdefault("low_confidence_format", "low_{person}_{original}")
    cfg.setdefault("low_confidence_threshold", 60)
    cfg.setdefault("use_cache", True)
    cfg.setdefault("cache_file", ".face_encodings_cache.pkl")
    cfg.setdefault("log_to_file", True)
    cfg.setdefault("log_file", "rename_log.txt")
    cfg.setdefault("exclude_dirs", [])

    if cfg.get("max_images_per_person") is not None:
        try:
            cfg["max_images_per_person"] = int(cfg["max_images_per_person"])
        except ValueError:
            cfg["max_images_per_person"] = 50

    return cfg


# ─────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────

def _compute_dir_fingerprint(person_dirs: list[Path], engine: str = "insightface", insightface_model: str = "buffalo_sc") -> str:
    """
    Tạo fingerprint dựa trên engine + model + mtime của folder.
    """
    hasher = hashlib.md5()
    hasher.update(f"engine:{engine}:model:{insightface_model}".encode())
    for d in sorted(person_dirs):
        dir_stat = d.stat()
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
    top_k: int = 3,
    margin_threshold: float = 0.04,
    engine: str = "insightface",
    insightface_model: str = "buffalo_sc",
    cosine_threshold: float = 0.38,
) -> tuple[str, str, float]:
    """
    Nhận diện khuôn mặt trong video bằng cách trích xuất khung hình và bầu chọn (Majority Voting).
    """
    frames = _extract_frames_from_video(video_path, num_frames=sample_frames)
    if not frames:
        return unknown_prefix, "không đọc được video / video trống", 0.0

    person_votes: dict[str, list[float]] = {}
    valid_frames_count = 0

    if engine == "insightface":
        app = get_insightface_app(insightface_model)
        if app is not None:
            import cv2
            for frame in frames:
                img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                faces = app.get(img_bgr)
                if not faces:
                    continue

                valid_frames_count += 1
                for face in faces:
                    emb = face.embedding
                    norm = np.linalg.norm(emb)
                    if norm == 0:
                        continue
                    face_enc = (emb / norm).astype(np.float32)

                    scores = {}
                    for person_name, known_encs in known_faces.items():
                        if len(known_encs) == 0:
                            continue
                        sims = np.dot(known_encs, face_enc)
                        k_highest = np.sort(sims)[-min(top_k, len(sims)):]
                        scores[person_name] = float(np.mean(k_highest))

                    if not scores:
                        continue

                    sorted_persons = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                    best_person, best_sim = sorted_persons[0]

                    if len(sorted_persons) > 1:
                        second_sim = sorted_persons[1][1]
                        if best_sim >= cosine_threshold and (best_sim - second_sim) < margin_threshold:
                            continue

                    if best_sim >= cosine_threshold:
                        conf = round(best_sim * 100, 1)
                        person_votes.setdefault(best_person, []).append(conf)

            if not person_votes:
                return unknown_prefix, f"video ({len(frames)} frames): không nhận diện được ai", 0.0

            top_person, confs = max(person_votes.items(), key=lambda item: (len(item[1]), np.mean(item[1])))
            avg_conf = round(float(np.mean(confs)), 1)
            votes_count = len(confs)

            reason = f"video: {votes_count}/{valid_frames_count} frames khớp {top_person} (conf trung bình {avg_conf}%)"
            return top_person, reason, avg_conf

    # Fallback to dlib video recognition
    for frame in frames:
        locations = face_recognition.face_locations(frame, model=model)
        if not locations:
            continue
        encodings = face_recognition.face_encodings(frame, locations)
        if not encodings:
            continue

        valid_frames_count += 1
        for face_enc in encodings:
            scores = {}
            for person_name, known_encs in known_faces.items():
                distances = face_recognition.face_distance(known_encs, face_enc)
                if len(distances) == 0:
                    continue
                k_smallest = np.sort(distances)[:min(top_k, len(distances))]
                scores[person_name] = float(np.mean(k_smallest))

            if not scores:
                continue

            sorted_persons = sorted(scores.items(), key=lambda item: item[1])
            best_person, best_distance = sorted_persons[0]

            if len(sorted_persons) > 1:
                second_distance = sorted_persons[1][1]
                if best_distance <= tolerance and (second_distance - best_distance) < margin_threshold:
                    continue

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
    num_jitters: int = 2,
    engine: str = "insightface",
    insightface_model: str = "buffalo_sc",
) -> dict[str, list]:
    """
    Quét tất cả folder con (trừ exclude_dirs), encode khuôn mặt,
    trả về dict {person_name: [encoding, ...]}.
    """
    exclude_set = {d.lower() for d in exclude_dirs}
    person_dirs = [
        d for d in dataset_root.iterdir()
        if d.is_dir() and d.name.lower() not in exclude_set
    ]

    if not person_dirs:
        logger.error("❌ Không tìm thấy folder person nào trong dataset_root.")
        sys.exit(1)

    logger.info(f"📂 Tìm thấy {len(person_dirs)} folder người: {[d.name for d in person_dirs]}")

    if use_cache:
        fingerprint = _compute_dir_fingerprint(person_dirs, engine=engine, insightface_model=insightface_model)
        cached = load_cache(cache_path, fingerprint, logger)
        if cached is not None:
            return cached

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

        shuffled = all_images.copy()
        random.shuffle(shuffled)

        encodings = []
        no_face_count = 0
        processed_count = 0
        target = max_images_per_person

        for img_path in tqdm(shuffled, desc=f"  Encoding {person_name} ({engine})", leave=False, unit="img"):
            if target and len(encodings) >= target:
                break

            enc = _encode_image(
                img_path, model, logger,
                num_jitters=num_jitters,
                engine=engine,
                insightface_model=insightface_model
            )
            processed_count += 1

            if enc:
                encodings.extend(enc)
            else:
                no_face_count += 1

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

    if use_cache:
        save_cache(cache_path, known_faces, fingerprint, logger)

    return known_faces


def _encode_image(
    img_path: Path,
    model: str,
    logger: logging.Logger,
    num_jitters: int = 2,
    engine: str = "insightface",
    insightface_model: str = "buffalo_sc"
) -> list:
    """Load ảnh (xử lý xoay EXIF) và trả về list encodings.
    
    InsightFace: Lọc face quality theo det_score >= 0.6 khi training.
    """
    try:
        pil_img = Image.open(img_path)
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        img_array = np.array(pil_img)

        if engine == "insightface":
            app = get_insightface_app(insightface_model)
            if app is not None:
                import cv2
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                faces = app.get(img_bgr)
                encs = []
                for face in faces:
                    # Lọc face chất lượng thấp trong training (det_score < 0.6)
                    det_score = getattr(face, 'det_score', 1.0)
                    if det_score < 0.6:
                        continue
                    emb = face.embedding
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        encs.append((emb / norm).astype(np.float32))
                return encs

        locations = face_recognition.face_locations(img_array, model=model)
        encs = face_recognition.face_encodings(img_array, locations, num_jitters=num_jitters)
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
    top_k: int = 3,
    margin_threshold: float = 0.04,
    engine: str = "insightface",
    insightface_model: str = "buffalo_l",
    cosine_threshold: float = 0.35,
    multi_face_strategy: str = "largest",
) -> tuple[str, str, float]:
    """
    Nhận diện khuôn mặt trong ảnh với InsightFace (ArcFace Cosine Similarity) hoặc dlib.
    multi_face_strategy: "largest" (chọn mặt lớn nhất) | "reject" (bỏ qua ảnh nhiều mặt)
    """
    try:
        pil_img = Image.open(img_path)
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        img_array = np.array(pil_img)
    except (UnidentifiedImageError, Exception) as e:
        return unknown_prefix, f"không đọc được ảnh ({e})", 0.0

    if engine == "insightface":
        app = get_insightface_app(insightface_model)
        if app is not None:
            import cv2
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            faces = app.get(img_bgr)

            if len(faces) == 0:
                return unknown_prefix, "không phát hiện khuôn mặt", 0.0

            # Xử lý nhiều khuôn mặt theo chiến lược
            if len(faces) > 1:
                if multi_face_strategy == "largest":
                    def _bbox_area(face) -> float:
                        bbox = face.bbox  # [x1, y1, x2, y2]
                        return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    best_face = max(faces, key=_bbox_area)
                    logger.debug(f"     ℹ️  {img_path.name}: {len(faces)} mặt → chọn mặt lớn nhất (det_score={best_face.det_score:.2f})")
                else:  # "reject"
                    return unknown_prefix, f"có {len(faces)} khuôn mặt trong ảnh (dùng multi_face_strategy=largest để tự chọn)", 0.0
            else:
                best_face = faces[0]

            emb = best_face.embedding
            norm = np.linalg.norm(emb)
            if norm == 0:
                return unknown_prefix, "không encode được khuôn mặt", 0.0
            face_enc = (emb / norm).astype(np.float32)

            scores = {}
            for person_name, known_encs in known_faces.items():
                if len(known_encs) == 0:
                    continue
                sims = np.dot(known_encs, face_enc)
                k_highest = np.sort(sims)[-min(top_k, len(sims)):]
                scores[person_name] = float(np.mean(k_highest))

            if not scores:
                return unknown_prefix, "không có known faces hợp lệ", 0.0

            sorted_persons = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            best_person, best_sim = sorted_persons[0]

            if len(sorted_persons) > 1:
                second_person, second_sim = sorted_persons[1]
                if best_sim >= cosine_threshold and (best_sim - second_sim) < margin_threshold:
                    return unknown_prefix, f"mập mờ giữa {best_person} ({best_sim:.3f}) và {second_person} ({second_sim:.3f})", 0.0

            confidence = round(best_sim * 100, 1)

            if best_sim >= cosine_threshold:
                return best_person, f"confidence {confidence}% (ArcFace sim={best_sim:.3f})", confidence
            else:
                return unknown_prefix, f"không khớp (ArcFace sim={best_sim:.3f} < threshold={cosine_threshold})", 0.0

    # Fallback to dlib
    locations = face_recognition.face_locations(img_array, model=model)
    if len(locations) == 0:
        return unknown_prefix, "không phát hiện khuôn mặt", 0.0
    if len(locations) > 1:
        return unknown_prefix, f"có {len(locations)} khuôn mặt trong ảnh", 0.0

    encodings = face_recognition.face_encodings(img_array, locations)
    if not encodings:
        return unknown_prefix, "không encode được khuôn mặt", 0.0

    face_enc = encodings[0]
    scores = {}
    for person_name, known_encs in known_faces.items():
        distances = face_recognition.face_distance(known_encs, face_enc)
        if len(distances) == 0:
            continue
        k_smallest = np.sort(distances)[:min(top_k, len(distances))]
        scores[person_name] = float(np.mean(k_smallest))

    if not scores:
        return unknown_prefix, "không có known faces hợp lệ", 0.0

    sorted_persons = sorted(scores.items(), key=lambda item: item[1])
    best_person, best_distance = sorted_persons[0]

    if len(sorted_persons) > 1:
        second_person, second_distance = sorted_persons[1]
        if best_distance <= tolerance and (second_distance - best_distance) < margin_threshold:
            return unknown_prefix, f"mập mờ giữa {best_person} ({best_distance:.3f}) và {second_person} ({second_distance:.3f})", 0.0

    if best_distance <= tolerance:
        confidence = round((1 - best_distance) * 100, 1)
        return best_person, f"confidence {confidence}% (top-{top_k} dist={best_distance:.3f})", confidence
    else:
        return unknown_prefix, f"không khớp (top-{top_k} dist={best_distance:.3f} > tolerance={tolerance})", 0.0


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

    low_threshold = cfg.get("low_confidence_threshold", 60)
    sample_frames = cfg.get("sample_frames_per_video", 12)
    top_k = cfg.get("top_k_neighbors", 3)
    margin_threshold = cfg.get("margin_threshold", 0.04)
    engine = cfg.get("engine", "insightface")
    insightface_model = cfg.get("insightface_model", "buffalo_l")
    cosine_threshold = cfg.get("cosine_threshold", 0.35)
    multi_face_strategy = cfg.get("multi_face_strategy", "largest")

    stats = {"recognized": 0, "low_confidence": 0, "unknown": 0, "skipped": 0, "error": 0}
    results = []

    for file_path in tqdm(media_files, desc="Đang nhận diện", unit="file"):
        original_name = file_path.name

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
                top_k=top_k,
                margin_threshold=margin_threshold,
                engine=engine,
                insightface_model=insightface_model,
                cosine_threshold=cosine_threshold,
            )
        else:
            person, reason, confidence = recognize_person(
                file_path,
                known_faces,
                cfg["tolerance"],
                cfg["model"],
                cfg["unknown_prefix"],
                logger,
                top_k=top_k,
                margin_threshold=margin_threshold,
                engine=engine,
                insightface_model=insightface_model,
                cosine_threshold=cosine_threshold,
                multi_face_strategy=multi_face_strategy,
            )

        is_unknown = person == cfg["unknown_prefix"]

        if is_unknown:
            name_fmt = cfg["name_format"]
        elif confidence < low_threshold:
            name_fmt = cfg.get("low_confidence_format", "low_{person}_{original}")
        else:
            name_fmt = cfg["name_format"]

        new_name = build_new_name(person, original_name, name_fmt, cfg["unknown_prefix"])
        new_path = file_path.parent / new_name

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
    "--apply", "-a",
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
    "--unclassified-dir", "-u", "-d", "--dir", "--target-dir",
    default=None,
    help="Tên hoặc đường dẫn thư mục cần xử lý (vd: 'other', 'chua_phan_loai').",
)
def main(config: str, apply: bool, clear_cache: bool, unclassified_dir: Optional[str]):
    """
    \b
    Face Recognition File Renamer
    ─────────────────────────────
    Nhận diện khuôn mặt và đổi tên ảnh trong thư mục chưa phân loại.
    """
    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)

    dataset_root = Path(cfg["dataset_root"]).expanduser().resolve()
    if not dataset_root.exists():
        click.echo(f"❌ dataset_root không tồn tại: {dataset_root}", err=True)
        sys.exit(1)

    target_str = unclassified_dir or cfg["unclassified_dir"]
    target_path = Path(target_str).expanduser().resolve()
    if target_path.exists() and target_path.is_dir():
        unclassified_path = target_path
    else:
        unclassified_path = dataset_root / target_str

    if not unclassified_path.exists():
        click.echo(f"❌ Thư mục không tồn tại: {unclassified_path}", err=True)
        sys.exit(1)

    exclude_dirs: list[str] = cfg["exclude_dirs"]
    if unclassified_path.name.lower() not in [d.lower() for d in exclude_dirs]:
        exclude_dirs.append(unclassified_path.name)

    cache_path = dataset_root / cfg["cache_file"]

    logger = setup_logger(cfg["log_to_file"], cfg["log_file"], dataset_root)

    logger.info("=" * 60)
    logger.info("  Face Recognition File Renamer")
    logger.info(f"  Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info(f"📌 Dataset root    : {dataset_root}")
    logger.info(f"📌 Unclassified    : {unclassified_path}")
    logger.info(f"📌 Exclude dirs    : {exclude_dirs}")
    logger.info(f"📌 Engine          : {cfg.get('engine', 'insightface')} ({cfg.get('insightface_model', 'buffalo_sc')})")
    logger.info(f"📌 Cosine Threshold: {cfg.get('cosine_threshold', 0.38)}")
    logger.info(f"📌 Max img/person  : {cfg['max_images_per_person']}")
    logger.info(f"📌 Cache           : {'on' if cfg['use_cache'] else 'off'}")
    logger.info(f"📌 Low conf (<%)   : {cfg.get('low_confidence_threshold', 60)}%")
    logger.info(f"📌 Mode            : {'APPLY ✏️' if apply else 'DRY-RUN 🔍'}")
    logger.info("")

    if clear_cache and cache_path.exists():
        cache_path.unlink()
        logger.info(f"🗑️  Đã xóa cache: {cache_path}")

    logger.info("📚 Loading known faces...")
    known_faces = load_known_faces(
        dataset_root=dataset_root,
        exclude_dirs=exclude_dirs,
        max_images_per_person=cfg["max_images_per_person"],
        model=cfg["model"],
        cache_path=cache_path,
        use_cache=cfg["use_cache"],
        logger=logger,
        num_jitters=cfg.get("num_jitters", 2),
        engine=cfg.get("engine", "insightface"),
        insightface_model=cfg.get("insightface_model", "buffalo_sc"),
    )
    logger.info(f"\n✅ Loaded {len(known_faces)} người: {list(known_faces.keys())}")
    total_encs = sum(len(v) for v in known_faces.values())
    logger.info(f"   Tổng {total_encs} face encodings\n")

    stats = process_unclassified(
        unclassified_dir=unclassified_path,
        known_faces=known_faces,
        cfg=cfg,
        apply=apply,
        logger=logger,
    )

    logger.info("\n" + "=" * 60)
    logger.info("  KẾT QUẢ")
    logger.info("=" * 60)
    logger.info(f"  ✅ Nhận diện được : {stats.get('recognized', 0)}")
    logger.info(f"  ⚠️  Low confidence : {stats.get('low_confidence', 0)}  (< {cfg.get('low_confidence_threshold', 60)}%)")
    logger.info(f"  ❓ Unknown        : {stats.get('unknown', 0)}")
    logger.info(f"  ⏭️  Đã xử lý trước : {stats.get('skipped', 0)}")
    logger.info(f"  ❌ Lỗi            : {stats.get('error', 0)}")
    if not apply:
        logger.info("\n  💡 Chạy với --apply để thực sự đổi tên.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
