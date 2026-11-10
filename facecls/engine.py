"""Trừu tượng hóa Engine nhận diện khuôn mặt (InsightFace & dlib) với cơ chế Fail-Fast."""

import logging
import threading
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from facecls.io import load_image_rgb, VALID_VIDEO_EXTENSIONS

logger = logging.getLogger("face_renamer")

_INSIGHTFACE_APP = None
_INSIGHTFACE_LOCK = threading.Lock()


def get_onnx_providers() -> List[str]:
    """Tự động phát hiện và ưu tiên GPU (CUDA) nếu có, fallback sang CPU."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        selected = []
        if "CUDAExecutionProvider" in available:
            selected.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            selected.append("CPUExecutionProvider")
        return selected or ["CPUExecutionProvider"]
    except Exception:
        return ["CPUExecutionProvider"]


def get_insightface_app(model_name: str = "buffalo_l", allowed_modules: Optional[List[str]] = None):
    """
    Khởi tạo InsightFace với cơ chế Singleton an toàn luồng.
    Tối ưu hóa tốc độ: CHỈ nạp các module cần thiết (mặc định: detection + recognition).
    """
    global _INSIGHTFACE_APP
    if _INSIGHTFACE_APP is not None:
        return _INSIGHTFACE_APP

    with _INSIGHTFACE_LOCK:
        if _INSIGHTFACE_APP is not None:
            return _INSIGHTFACE_APP

        try:
            import warnings
            import insightface
            from insightface.app import FaceAnalysis

            logging.getLogger("insightface").setLevel(logging.WARNING)
            warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

            modules = allowed_modules or ["detection", "recognition"]
            providers = get_onnx_providers()
            logger.info(f"🚀 Khởi tạo InsightFace (model={model_name}, modules={modules}, providers={providers})...")

            app = FaceAnalysis(name=model_name, allowed_modules=modules, providers=providers)
            app.prepare(ctx_id=0, det_size=(640, 640))
            _INSIGHTFACE_APP = app
            return _INSIGHTFACE_APP
        except Exception as e:
            raise RuntimeError(f"❌ Không thể khởi tạo InsightFace ({model_name}): {e}") from e


class FaceEngine:
    """Quản lý trích xuất vector đặc trưng khuôn mặt (Embedding)."""

    def __init__(
        self,
        engine: str = "insightface",
        insightface_model: str = "buffalo_l",
        allowed_modules: Optional[List[str]] = None,
        det_thresh: float = 0.6,
        multi_face_strategy: str = "largest",
        model: str = "hog",
        num_jitters: int = 2,
        image_max_size: int = 1600,
    ):
        self.engine = engine.lower()
        self.insightface_model = insightface_model
        self.allowed_modules = allowed_modules or ["detection", "recognition"]
        self.det_thresh = det_thresh
        self.multi_face_strategy = multi_face_strategy
        self.model = model
        self.num_jitters = num_jitters
        self.image_max_size = image_max_size

        # Kiểm tra tính sẵn sàng của engine ngay khi khởi tạo (Fail-Fast)
        if self.engine == "insightface":
            self.app = get_insightface_app(self.insightface_model, self.allowed_modules)
        elif self.engine == "dlib":
            try:
                import face_recognition
                self.fr = face_recognition
            except ImportError as e:
                raise RuntimeError("❌ Không tìm thấy thư viện face_recognition cho dlib engine.") from e
        else:
            raise ValueError(f"❌ Engine không hỗ trợ: '{self.engine}'. Chọn 'insightface' hoặc 'dlib'.")

    def extract_face_embeddings(
        self,
        img_rgb: np.ndarray,
        filter_quality: bool = True,
        select_single: bool = True,
        det_thresh: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Trích xuất danh sách các vector khuôn mặt từ ảnh RGB.
        select_single=True: áp dụng multi_face_strategy ("largest" hoặc "reject").
        det_thresh: tuỳ chọn ghi đè ngưỡng phát hiện khuôn mặt (hữu ích cho video motion blur).
        """
        if img_rgb is None or img_rgb.size == 0:
            return []

        if self.engine == "insightface":
            # InsightFace nhận định dạng BGR
            img_bgr = img_rgb[:, :, ::-1]
            faces = self.app.get(img_bgr)
            if not faces:
                return []

            # Lọc theo chất lượng phát hiện khuôn mặt (det_score)
            active_thresh = det_thresh if det_thresh is not None else self.det_thresh
            if filter_quality and active_thresh > 0:
                faces = [f for f in faces if getattr(f, "det_score", 1.0) >= active_thresh]

            if not faces:
                return []

            # Xử lý ảnh có nhiều khuôn mặt
            if select_single and len(faces) > 1:
                if self.multi_face_strategy == "largest":
                    # Chọn khuôn mặt có diện tích bbox lớn nhất
                    faces = [max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))]
                elif self.multi_face_strategy == "reject":
                    return []

            embeddings = []
            for f in faces:
                emb = f.embedding
                norm = np.linalg.norm(emb)
                if norm > 0:
                    embeddings.append((emb / norm).astype(np.float32))
            return embeddings

        # Engine dlib
        locations = self.fr.face_locations(img_rgb, model=self.model)
        if not locations:
            return []

        if select_single and len(locations) > 1:
            if self.multi_face_strategy == "largest":
                locations = [max(locations, key=lambda loc: float((loc[2] - loc[0]) * (loc[1] - loc[3])))]
            elif self.multi_face_strategy == "reject":
                return []

        encs = self.fr.face_encodings(img_rgb, locations, num_jitters=self.num_jitters)
        return [np.asarray(e, dtype=np.float32) for e in encs]

    def extract_faces_with_bboxes(
        self,
        img_rgb: np.ndarray,
        filter_quality: bool = True,
        det_thresh: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Trích xuất danh sách khuôn mặt kèm toạ độ bounding box (tỷ lệ %) và embedding.
        """
        if img_rgb is None or img_rgb.size == 0:
            return []

        h, w = img_rgb.shape[:2]
        if h <= 0 or w <= 0:
            return []

        results = []
        if self.engine == "insightface":
            img_bgr = img_rgb[:, :, ::-1]
            faces = self.app.get(img_bgr) or []
            active_thresh = det_thresh if det_thresh is not None else self.det_thresh
            if filter_quality and active_thresh > 0:
                faces = [f for f in faces if getattr(f, "det_score", 1.0) >= active_thresh]

            for f in faces:
                b = f.bbox
                emb = f.embedding
                norm = np.linalg.norm(emb)
                face_enc = (emb / norm).astype(np.float32) if norm > 0 else None
                score = round(float(getattr(f, "det_score", 1.0)) * 100, 1)
                bbox_dict = {
                    "x": float(b[0] / w),
                    "y": float(b[1] / h),
                    "w": float((b[2] - b[0]) / w),
                    "h": float((b[3] - b[1]) / h),
                    "score": score,
                }
                results.append({
                    "bbox": bbox_dict,
                    "embedding": face_enc,
                    "det_score": getattr(f, "det_score", 1.0),
                })
        else:
            locations = self.fr.face_locations(img_rgb, model=self.model) or []
            encs = self.fr.face_encodings(img_rgb, locations, num_jitters=self.num_jitters) if locations else []
            for loc, enc in zip(locations, encs):
                top, right, bottom, left = loc
                bbox_dict = {
                    "x": float(left / w),
                    "y": float(top / h),
                    "w": float((right - left) / w),
                    "h": float((bottom - top) / h),
                    "score": 100.0,
                }
                results.append({
                    "bbox": bbox_dict,
                    "embedding": np.asarray(enc, dtype=np.float32),
                    "det_score": 1.0,
                })

        return results

    def extract_from_file(
        self,
        file_path: Path,
        filter_quality: bool = True,
        select_single: bool = True
    ) -> List[np.ndarray]:
        """Trích xuất embedding từ tệp ảnh."""
        img_rgb = load_image_rgb(file_path, max_size=self.image_max_size)
        if img_rgb is None:
            return []
        return self.extract_face_embeddings(
            img_rgb, filter_quality=filter_quality, select_single=select_single
        )

    def extract_frames_from_video(
        self,
        video_path: Union[str, Path],
        sample_frames: int = 12,
        max_size: Optional[int] = 1080
    ) -> List[np.ndarray]:
        """
        Trích xuất danh sách khung hình RGB trải đều từ tệp video.
        Hỗ trợ an toàn các định dạng video web, streaming, mp4 không có sẵn frame count index.
        """
        try:
            import cv2
        except ImportError:
            return []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        def _format_frame(bgr: np.ndarray) -> np.ndarray:
            if max_size and max_size > 0:
                fh, fw = bgr.shape[:2]
                if max(fh, fw) > max_size:
                    scale = max_size / max(fh, fw)
                    bgr = cv2.resize(bgr, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
            return bgr[:, :, ::-1]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: List[np.ndarray] = []

        # Chiến lược 1: Thử seeking nếu tổng số frame hợp lệ
        if total_frames > 0:
            target_indices = [
                min(int(i * total_frames / sample_frames), total_frames - 1)
                for i in range(sample_frames)
            ]
            for idx in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    frames.append(_format_frame(frame))

        # Chiến lược 2: Nếu frame_count <= 0 hoặc seeking chỉ lấy được quá ít frame
        if len(frames) < min(3, max(1, sample_frames // 2)):
            frames.clear()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            all_read = []
            while True:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    break
                all_read.append(frame)
                # Giới hạn an toàn bộ nhớ: không nạp quá 1200 frame thô vào RAM
                if len(all_read) >= 1200:
                    break

            if all_read:
                n = len(all_read)
                step = max(1, n // sample_frames)
                indices = [min(i * step, n - 1) for i in range(min(sample_frames, n))]
                for idx in indices:
                    frames.append(_format_frame(all_read[idx]))

        cap.release()
        return frames
