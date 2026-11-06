"""Module trích xuất và lọc khuôn mặt từ Video, Reels, Stories và Highlights.

Hỗ trợ lấy mẫu frame định kỳ, kiểm tra độ tương đồng ArcFace theo mỏ neo danh tính,
khử trùng lặp tư thế và lưu trữ ảnh frame chất lượng cao.
"""

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("face_renamer")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}

VIDEO_URL_PATTERNS = [
    r"facebook\.com/.+/videos/",
    r"facebook\.com/watch",
    r"facebook\.com/reel/",
    r"facebook\.com/stories/",
    r"fb\.watch/",
    r"instagram\.com/reel/",
    r"instagram\.com/stories/highlights/",
    r"instagram\.com/stories/",
    r"instagram\.com/p/[^/]+/\?.*video",
    r"tiktok\.com/",
    r"youtube\.com/shorts/",
    r"youtu\.be/",
]


def is_video_url(url_or_path: str) -> bool:
    """Kiểm tra một đường link hoặc đường dẫn có phải là video/highlight/reel hay không."""
    cleaned = url_or_path.strip().lower()
    for ext in VIDEO_EXTENSIONS:
        if cleaned.endswith(ext) or f"{ext}?" in cleaned:
            return True
    for pat in VIDEO_URL_PATTERNS:
        if re.search(pat, cleaned):
            return True
    return False


def download_video_stream(
    url: str,
    temp_dir: Optional[Path] = None,
    use_browser_cookies: bool = True,
) -> Optional[Path]:
    """Tải video hoặc highlight từ URL về thư mục tạm bằng yt-dlp hoặc trực tiếp từ CDN."""
    import yt_dlp

    t_dir = temp_dir or Path(tempfile.mkdtemp(prefix="facecls_vid_"))
    t_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(t_dir / "%(id)s.%(ext)s")

    # Nếu là URL CDN trực tiếp (.mp4 hoặc fbcdn/instagram cdn), tải trực tiếp bằng urllib cực nhanh
    if url.startswith("http") and (".mp4" in url.lower() or "fbcdn.net" in url.lower() or "cdninstagram" in url.lower()):
        try:
            import urllib.request
            fast_p = t_dir / f"direct_clip_{os.urandom(4).hex()}.mp4"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                if len(data) > 1000:
                    fast_p.write_bytes(data)
                    return fast_p
        except Exception as e:
            logger.debug(f"Direct stream download failed, fallback to yt-dlp: {e}")

    def _try_download(cookies_browser: Optional[str]) -> Optional[Path]:
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 15,
        }
        if cookies_browser:
            ydl_opts["cookiesfrombrowser"] = (cookies_browser, None, None, None)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    return None
                filename = ydl.prepare_filename(info)
                p = Path(filename)
                if p.exists() and p.stat().st_size > 1000:
                    return p
                # Trường hợp yt-dlp gộp định dạng thành file khác
                for f in t_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS and f.stat().st_size > 1000:
                        return f
        except Exception as e:
            logger.debug(f"yt-dlp download failed with browser={cookies_browser}: {e}")
            return None
        return None

    # Thử với Brave cookies trước (nếu bật), sau đó thử trực tiếp không cookies
    res = None
    if use_browser_cookies:
        res = _try_download("brave")
    if res is None:
        res = _try_download(None)

    return res


def analyze_video_motion_and_best_frame(
    video_path: Path,
    num_samples: int = 5,
    engine_obj=None,
) -> Tuple[bool, Optional[np.ndarray], str]:
    """
    Phân tích độ chuyển động của video để phát hiện xem video có phải là video ảnh tĩnh
    (chỉ có 1 kiểu ảnh duy nhất ghép nhạc nền / audio / lyric / visualizer) hay không.

    Trả về:
        (is_static, best_frame_rgb, reason)
        - is_static (bool): True nếu video là ảnh tĩnh (cần bỏ qua lưu video và chỉ chụp 1 frame ảnh).
        - best_frame_rgb (np.ndarray): Khung hình rõ nét nhất (chống nhòe, sáng đẹp) để lưu thành ảnh .jpg.
        - reason (str): Lý giải kết quả phân tích.
    """
    if not video_path.exists():
        return False, None, "File không tồn tại"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, None, "Không thể đọc video"

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if total_frames <= 1:
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return True, rgb, "Video chỉ có đúng 1 frame ảnh tĩnh"
            return False, None, "Không trích xuất được frame"

        # Lấy mẫu các vị trí rải đều theo thời lượng video (tránh 5% đầu/cuối hay bị fade in/out)
        ratios = [0.15, 0.35, 0.55, 0.75, 0.90] if num_samples >= 5 else [0.2, 0.5, 0.8]
        sample_indices = sorted(list(set([max(0, min(total_frames - 1, int(total_frames * r))) for r in ratios])))
        if len(sample_indices) < 2:
            sample_indices = [0, max(0, total_frames - 1)]

        sampled_frames: List[Tuple[int, np.ndarray]] = []
        for s_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, s_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append((s_idx, rgb))

        if not sampled_frames:
            return False, None, "Không đọc được khung hình nào từ video"

        if len(sampled_frames) == 1:
            return True, sampled_frames[0][1], "Chỉ trích xuất được 1 khung hình duy nhất"

        # Đánh giá độ sắc nét và ánh sáng để chọn frame tối ưu nhất
        def _frame_quality(f_rgb: np.ndarray, is_edge: bool = False) -> float:
            gray = cv2.cvtColor(f_rgb, cv2.COLOR_RGB2GRAY)
            b = float(np.mean(gray))
            if b < 30 or b > 240:
                return -10.0
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            return sharp * (0.8 if is_edge else 1.0)

        best_frame = max(
            sampled_frames,
            key=lambda item: _frame_quality(item[1], is_edge=(item[0] in [0, total_frames - 1]))
        )[1]

        # 1. Thu nhỏ để so sánh độ lệch pixel tổng thể nhanh chóng (160x160)
        grays = [cv2.resize(cv2.cvtColor(f[1], cv2.COLOR_RGB2GRAY), (160, 160)) for f in sampled_frames]
        diff_0_last = cv2.absdiff(grays[0], grays[-1])
        overall_mean = float(np.mean(diff_0_last))

        diff_0_mid = cv2.absdiff(grays[0], grays[len(grays) // 2])
        mid_mean = float(np.mean(diff_0_mid))

        # Tiêu chí 1: Ảnh tĩnh tuyệt đối (toàn bộ video gần như không đổi qua thời gian)
        if overall_mean < 4.0 and mid_mean < 4.0:
            return True, best_frame, f"Ảnh tĩnh tuyệt đối (độ lệch khung hình {overall_mean:.2f})"

        # Tiêu chí 2: Ảnh tĩnh có sóng nhạc, chữ chạy (lyrics) hoặc thanh thời gian
        # Chia thành lưới 8x8 = 64 ô để đếm tỷ lệ các ô tĩnh
        block_h, block_w = 20, 20
        static_blocks = 0
        central_static = 0
        central_total = 0

        for r in range(8):
            for c in range(8):
                blk = diff_0_last[r * block_h : (r + 1) * block_h, c * block_w : (c + 1) * block_w]
                is_blk_static = (np.mean(blk) < 4.5)
                if is_blk_static:
                    static_blocks += 1
                if 1 <= r <= 6 and 1 <= c <= 6:
                    central_total += 1
                    if is_blk_static:
                        central_static += 1

        static_ratio = static_blocks / 64.0
        central_ratio = central_static / max(1, central_total)

        if static_ratio >= 0.70 and central_ratio >= 0.75:
            return True, best_frame, f"Ảnh tĩnh có lớp chữ/nhạc ({static_ratio * 100:.0f}% diện tích tĩnh)"

        # Tiêu chí 3: Ảnh tĩnh có hiệu ứng zoom nhẹ (Ken Burns) dựa trên khuôn mặt
        if engine_obj is not None:
            try:
                f0_rgb = sampled_frames[0][1]
                fn_rgb = sampled_frames[-1][1]
                e0 = engine_obj.extract_face_embeddings(f0_rgb, filter_quality=True, select_single=True)
                en = engine_obj.extract_face_embeddings(fn_rgb, filter_quality=True, select_single=True)
                if e0 and en:
                    face_sim = float(np.dot(e0[0], en[0]))
                    if face_sim >= 0.985 and overall_mean < 32.0:
                        return True, best_frame, f"Ảnh tĩnh có hiệu ứng zoom nhẹ (khớp mặt 100%, sim={face_sim * 100:.1f}%)"
            except Exception:
                pass

        return False, best_frame, f"Video chuyển động thực sự (độ lệch {overall_mean:.1f})"
    finally:
        cap.release()


def sample_video_frames(
    video_path: Path,
    sample_interval_sec: float = 0.75,
    max_frames: int = 40,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Trích xuất các frame hình ảnh từ file video theo khoảng thời gian chỉ định (giây).
    Yields (frame_index, timestamp_sec, rgb_frame_array).
    """
    if not video_path.exists():
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.debug(f"Không thể mở file video: {video_path}")
        return

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step_frames = max(1, int(fps * sample_interval_sec))

        current_frame = 0
        yielded_count = 0

        while current_frame < total_frames and yielded_count < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = current_frame / fps
            yield (current_frame, timestamp, rgb)

            yielded_count += 1
            current_frame += step_frames
    finally:
        cap.release()


def process_video_and_extract_faces(
    video_path: Path,
    output_dir: Path,
    clean_prefix: str,
    engine_obj,
    anchor_centroid: Optional[np.ndarray] = None,
    similarity_threshold: float = 0.38,
    sample_interval_sec: float = 0.75,
    max_saved: int = 15,
) -> Tuple[int, Optional[np.ndarray]]:
    """
    Quét video, cắt frame và nhận diện người mỏ neo, tự động lưu các frame đạt chuẩn và khử trùng lặp tư thế.
    Trả về (số frame đã lưu, anchor_centroid_mới).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_files = [f.stem for f in output_dir.iterdir() if f.is_file()]
    idx = 1
    while f"{clean_prefix}_{idx:04d}" in existing_files:
        idx += 1

    saved_count = 0
    saved_embeddings: List[np.ndarray] = []
    current_anchor = anchor_centroid

    for frame_idx, timestamp_sec, rgb_frame in sample_video_frames(
        video_path,
        sample_interval_sec=sample_interval_sec,
        max_frames=max_saved * 4,
    ):
        if saved_count >= max_saved:
            break

        embs = engine_obj.extract_face_embeddings(rgb_frame, filter_quality=True, select_single=True)
        if not embs:
            continue

        face_enc = embs[0]

        # Khởi tạo mỏ neo nếu chưa có
        if current_anchor is None:
            current_anchor = face_enc
        else:
            sim = float(np.dot(current_anchor, face_enc))
            if sim < similarity_threshold:
                continue

            # Khử trùng lặp frame: nếu góc mặt quá giống frame trước đó (> 0.92) thì bỏ qua
            is_duplicate = False
            for prev_emb in saved_embeddings[-3:]:
                if float(np.dot(prev_emb, face_enc)) > 0.92:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            # Cập nhật mỏ neo
            current_anchor = 0.95 * current_anchor + 0.05 * face_enc
            current_anchor = current_anchor / np.linalg.norm(current_anchor)

        # Lưu ảnh frame đạt chuẩn
        filename = f"{clean_prefix}_{idx:04d}.jpg"
        save_path = output_dir / filename
        while save_path.exists():
            idx += 1
            filename = f"{clean_prefix}_{idx:04d}.jpg"
            save_path = output_dir / filename

        try:
            img = Image.fromarray(rgb_frame)
            img.save(save_path, "JPEG", quality=92)
            saved_count += 1
            saved_embeddings.append(face_enc)
            idx += 1
        except Exception as e:
            logger.debug(f"Không thể lưu frame video {save_path}: {e}")

    return saved_count, current_anchor
