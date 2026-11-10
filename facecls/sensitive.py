"""Module tích hợp NudeNet AI phát hiện nội dung nhạy cảm (NSFW / Sensitive)."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

logger = logging.getLogger("face_sensitive")

EXPLICIT_CATEGORIES = {
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

SENSITIVE_CATEGORIES = {
    "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_COVERED",
    "BUTTOCKS_COVERED",
    "MALE_BREAST_EXPOSED",
}

CATEGORY_LABELS_VI = {
    "FEMALE_GENITALIA_EXPOSED": "Vùng kín nữ (Lộ)",
    "FEMALE_BREAST_EXPOSED": "Ngực nữ (Lộ)",
    "MALE_GENITALIA_EXPOSED": "Vùng kín nam (Lộ)",
    "BUTTOCKS_EXPOSED": "Mông (Lộ)",
    "ANUS_EXPOSED": "Hậu môn",
    "FEMALE_BREAST_COVERED": "Ngực nữ (Bikini/Đồ lót)",
    "FEMALE_GENITALIA_COVERED": "Vùng kín (Bikini/Đồ lót)",
    "BUTTOCKS_COVERED": "Mông (Bikini/Đồ lót)",
    "MALE_BREAST_EXPOSED": "Ngực nam trần",
    "FACE_FEMALE": "Khuôn mặt nữ",
    "FACE_MALE": "Khuôn mặt nam",
    "BELLY_EXPOSED": "Bụng trần",
    "BELLY_COVERED": "Bụng che",
    "FEET_EXPOSED": "Bàn chân",
    "ARMPITS_EXPOSED": "Nách trần",
}

CATEGORY_COLORS = {
    "FEMALE_GENITALIA_EXPOSED": "#ef4444",
    "FEMALE_BREAST_EXPOSED": "#f97316",
    "MALE_GENITALIA_EXPOSED": "#dc2626",
    "BUTTOCKS_EXPOSED": "#eab308",
    "ANUS_EXPOSED": "#b91c1c",
    "FEMALE_BREAST_COVERED": "#3b82f6",
    "FEMALE_GENITALIA_COVERED": "#6366f1",
    "BUTTOCKS_COVERED": "#8b5cf6",
    "MALE_BREAST_EXPOSED": "#06b6d4",
}

_SHARED_DETECTOR = None


def get_nudenet_detector():
    """Khởi tạo lazy singleton đối tượng NudeDetector."""
    global _SHARED_DETECTOR
    if _SHARED_DETECTOR is None:
        try:
            from nudenet import NudeDetector
            logger.info("Khởi tạo mô hình NudeNet ONNX (320n.onnx)...")
            _SHARED_DETECTOR = NudeDetector()
            logger.info("Nạp NudeDetector thành công.")
        except Exception as e:
            logger.error(f"Không thể khởi tạo NudeDetector: {e}")
            _SHARED_DETECTOR = None
    return _SHARED_DETECTOR


def check_sensitive(
    image_input: Union[str, Path, np.ndarray],
    threshold: float = 0.35,
) -> Dict[str, Any]:
    """
    Kiểm tra ảnh xem có chứa nội dung nhạy cảm / NSFW hay không.
    Trả về thông tin phân loại: is_sensitive, status, max_category, max_score, v.v.
    Hỗ trợ cả toạ độ pixel box [x, y, w, h] và toạ độ phần trăm box_pct {x, y, w, h}.
    """
    detector = get_nudenet_detector()
    if detector is None:
        return {
            "is_sensitive": False,
            "status": "SAFE",
            "max_category": None,
            "max_category_vi": None,
            "max_score": 0.0,
            "explicit_count": 0,
            "sensitive_count": 0,
            "detections": [],
            "img_width": 0,
            "img_height": 0,
        }

    # Lấy kích thước ảnh để tính tỷ lệ box_pct (%)
    img_w, img_h = 0, 0
    if isinstance(image_input, np.ndarray):
        img_h, img_w = image_input.shape[:2]
    elif isinstance(image_input, (str, Path)):
        try:
            from PIL import Image
            with Image.open(str(image_input)) as im:
                img_w, img_h = im.size
        except Exception:
            img_w, img_h = 0, 0

    try:
        target = str(image_input) if isinstance(image_input, Path) else image_input
        raw_detections = detector.detect(target) or []
    except Exception as e:
        logger.warning(f"Lỗi khi chạy NudeDetector: {e}")
        return {
            "is_sensitive": False,
            "status": "SAFE",
            "max_category": None,
            "max_category_vi": None,
            "max_score": 0.0,
            "explicit_count": 0,
            "sensitive_count": 0,
            "detections": [],
            "img_width": img_w,
            "img_height": img_h,
        }

    explicit_items = []
    sensitive_items = []
    filtered_detections = []

    for det in raw_detections:
        cls_name = det.get("class", "UNKNOWN")
        score = float(det.get("score", 0.0))
        box = det.get("box", [0, 0, 0, 0])

        if cls_name in EXPLICIT_CATEGORIES:
            cat_type = "explicit"
        elif cls_name in SENSITIVE_CATEGORIES:
            cat_type = "sensitive"
        else:
            cat_type = "neutral"

        if score >= threshold and cat_type in ("explicit", "sensitive"):
            box_pct = None
            if img_w > 0 and img_h > 0 and len(box) >= 4:
                box_pct = {
                    "x": round(float(box[0] / img_w) * 100, 2),
                    "y": round(float(box[1] / img_h) * 100, 2),
                    "w": round(float(box[2] / img_w) * 100, 2),
                    "h": round(float(box[3] / img_h) * 100, 2),
                }

            item = {
                "class": cls_name,
                "label_vi": CATEGORY_LABELS_VI.get(cls_name, cls_name),
                "score": round(score * 100, 1),
                "box": box,
                "box_pct": box_pct,
                "type": cat_type,
                "color": CATEGORY_COLORS.get(cls_name, "#a855f7"),
            }
            filtered_detections.append(item)
            if cat_type == "explicit":
                explicit_items.append(item)
            elif cat_type == "sensitive":
                sensitive_items.append(item)

    if explicit_items:
        max_item = max(explicit_items, key=lambda x: x["score"])
        is_sensitive = True
        status = "UNSAFE"
        max_category = max_item["class"]
        max_category_vi = max_item["label_vi"]
        max_score = max_item["score"]
    elif sensitive_items:
        max_item = max(sensitive_items, key=lambda x: x["score"])
        is_sensitive = True
        status = "SENSITIVE"
        max_category = max_item["class"]
        max_category_vi = max_item["label_vi"]
        max_score = max_item["score"]
    else:
        is_sensitive = False
        status = "SAFE"
        max_category = None
        max_category_vi = None
        max_score = 0.0

    return {
        "is_sensitive": is_sensitive,
        "status": status,
        "max_category": max_category,
        "max_category_vi": max_category_vi,
        "max_score": max_score,
        "explicit_count": len(explicit_items),
        "sensitive_count": len(sensitive_items),
        "detections": filtered_detections,
        "img_width": img_w,
        "img_height": img_h,
    }


def check_sensitive_media(
    media_input: Union[str, Path, np.ndarray],
    frames: Optional[List[np.ndarray]] = None,
    threshold: float = 0.35,
    max_video_frames: int = 5,
) -> Dict[str, Any]:
    """
    Kiểm tra nội dung nhạy cảm cho cả hình ảnh và video.
    Với video, lấy mẫu các khung hình và tổng hợp kết quả vi phạm cao nhất.
    """
    if frames is not None and len(frames) > 0:
        # Chọn mẫu đều các khung hình từ danh sách frame đã trích xuất
        step = max(1, len(frames) // max_video_frames)
        sampled = [frames[i * step] for i in range(min(max_video_frames, len(frames)))]
        
        all_detections = []
        is_any_sensitive = False
        highest_score = 0.0
        best_category = None
        best_category_vi = None
        status = "SAFE"
        explicit_cnt = 0
        sensitive_cnt = 0

        for frame in sampled:
            res = check_sensitive(frame, threshold=threshold)
            if res.get("is_sensitive"):
                is_any_sensitive = True
                explicit_cnt += res.get("explicit_count", 0)
                sensitive_cnt += res.get("sensitive_count", 0)
                if res.get("max_score", 0.0) > highest_score:
                    highest_score = res["max_score"]
                    best_category = res["max_category"]
                    best_category_vi = res["max_category_vi"]
                    status = res["status"]
            all_detections.extend(res.get("detections", []))

        return {
            "is_sensitive": is_any_sensitive,
            "status": status,
            "max_category": best_category,
            "max_category_vi": best_category_vi,
            "max_score": highest_score,
            "explicit_count": explicit_cnt,
            "sensitive_count": sensitive_cnt,
            "detections": all_detections,
            "is_video": True,
        }

    # Nếu media_input là tệp video và chưa có frames
    target_path = Path(media_input) if isinstance(media_input, (str, Path)) else None
    if target_path and target_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        try:
            import cv2
            cap = cv2.VideoCapture(str(target_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            extracted_frames = []
            if total > 0:
                indices = [int(i * total / max_video_frames) for i in range(max_video_frames)]
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, fr = cap.read()
                    if ret and fr is not None:
                        extracted_frames.append(fr[:, :, ::-1])
            else:
                count = 0
                while count < max_video_frames:
                    ret, fr = cap.read()
                    if not ret or fr is None:
                        break
                    extracted_frames.append(fr[:, :, ::-1])
                    count += 1
            cap.release()
            if extracted_frames:
                return check_sensitive_media(media_input, frames=extracted_frames, threshold=threshold)
        except Exception as e:
            logger.warning(f"Lỗi trích xuất video để kiểm tra nhạy cảm: {e}")

    # Mặc định kiểm tra như ảnh đơn
    return check_sensitive(media_input, threshold=threshold)
