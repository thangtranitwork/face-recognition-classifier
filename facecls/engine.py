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
        select_single: bool = True
    ) -> List[np.ndarray]:
        """
        Trích xuất danh sách các vector khuôn mặt từ ảnh RGB.
        select_single=True: áp dụng multi_face_strategy ("largest" hoặc "reject").
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
            if filter_quality and self.det_thresh > 0:
                faces = [f for f in faces if getattr(f, "det_score", 1.0) >= self.det_thresh]

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

    def extract_frames_from_video(self, video_path: Path, sample_frames: int = 12) -> List[np.ndarray]:
        """Trích xuất danh sách khung hình trải đều từ tệp video."""
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

        step = max(1, total_frames // sample_frames)
        frame_indices = [i * step for i in range(min(sample_frames, total_frames))]

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                # Đổi BGR sang RGB
                rgb_frame = frame[:, :, ::-1]
                frames.append(rgb_frame)

        cap.release()
        return frames
