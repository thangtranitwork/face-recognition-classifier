#!/usr/bin/env python3
"""
Auto-Clustering Unknown Faces Script
====================================
Tự động quét các ảnh/video chưa nhận diện (`unknown_*` hoặc chưa phân loại)
trong thư mục `chua_phan_loai/`, dùng thuật toán DBSCAN gom nhóm các khuôn mặt
của cùng 1 người lạ xuất hiện nhiều lần.

Tự động gợi ý và tạo folder người mới (ví dụ: `nguoi_moi_1`, `nguoi_moi_2`)
rồi di chuyển các ảnh đó vào folder mới để làm data training sẵn sàng!

Cách dùng:
  python cluster_faces.py              # Dry-run: Phân tích & xem trước các nhóm người mới phát hiện
  python cluster_faces.py --apply      # Thực sự tạo folder người mới và di chuyển file
"""

import sys
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

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

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    print("❌ Thiếu thư viện scikit-learn. Chạy: pip install scikit-learn")
    sys.exit(1)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALL_VALID_EXTENSIONS = VALID_EXTENSIONS | VALID_VIDEO_EXTENSIONS


def _extract_encoding_from_file(file_path: Path, model: str, sample_frames: int) -> Optional[np.ndarray]:
    """Trích xuất 1 encoding đại diện từ ảnh hoặc video."""
    ext = file_path.suffix.lower()

    if ext in VALID_VIDEO_EXTENSIONS:
        try:
            import cv2
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                return None
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return None

            step = max(1, total // sample_frames)
            for idx in [i * step for i in range(min(sample_frames, total))]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    locs = face_recognition.face_locations(rgb, model=model)
                    if locs:
                        encs = face_recognition.face_encodings(rgb, locs)
                        if encs:
                            cap.release()
                            return encs[0]
            cap.release()
        except Exception:
            return None
    else:
        try:
            pil_img = Image.open(file_path).convert("RGB")
            img_arr = np.array(pil_img)
            locs = face_recognition.face_locations(img_arr, model=model)
            if locs:
                encs = face_recognition.face_encodings(img_arr, locs)
                if encs:
                    return encs[0]
        except Exception:
            return None

    return None


def resolve_target_dir(target_dir_str: Optional[str], default_name: str, dataset_root: Path) -> Path:
    if target_dir_str:
        p = Path(target_dir_str).expanduser().resolve()
        if p.exists() and p.is_dir():
            return p
        rel_p = dataset_root / target_dir_str
        if rel_p.exists() and rel_p.is_dir():
            return rel_p
        click.echo(f"❌ Thư mục không tồn tại: '{target_dir_str}' (đã tìm ở {p} và {rel_p})", err=True)
        sys.exit(1)
    return dataset_root / default_name


@click.command()
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn file config YAML.")
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Thư mục cần gom nhóm (vd: 'other', 'chua_phan_loai' hoặc đường dẫn bất kỳ).")
@click.option("--apply", is_flag=True, default=False, help="Thực sự tạo folder người mới và di chuyển file.")
@click.option("--eps", default=None, type=float, help="Override ngưỡng khoảng cách DBSCAN (default trong config).")
@click.option("--min-samples", default=None, type=int, help="Override số ảnh tối thiểu/nhóm (default trong config).")
def main(config: str, target_dir: Optional[str], apply: bool, eps: Optional[float], min_samples: Optional[int]):
    cfg_path = Path(config)
    if not cfg_path.exists():
        click.echo(f"❌ Không tìm thấy file config: {config}", err=True)
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("dataset_root", "./dataset")).expanduser().resolve()
    unclassified_name = cfg.get("unclassified_dir", "chua_phan_loai")
    unclassified_dir = resolve_target_dir(target_dir, unclassified_name, dataset_root)
    model = cfg.get("model", "hog")
    sample_frames = cfg.get("sample_frames_per_video", 12)
    unknown_prefix = cfg.get("unknown_prefix", "unknown").lower()
    prefix_folder = cfg.get("cluster_prefix", "nguoi_moi")

    dbscan_eps = eps if eps is not None else float(cfg.get("cluster_eps", 0.45))
    dbscan_min_samples = min_samples if min_samples is not None else int(cfg.get("cluster_min_samples", 2))

    # Lấy danh sách tên những người đã biết trong dataset
    exclude_dirs = {d.lower() for d in cfg.get("exclude_dirs", [])}
    exclude_dirs.add(unclassified_dir.name.lower())
    known_person_names = {
        d.name.lower() for d in dataset_root.iterdir()
        if d.is_dir() and d.name.lower() not in exclude_dirs
    }

    # 1. Chỉ thu thập các file CHƯA phân loại (file unknown_ hoặc chưa được gán nhãn người đã biết)
    files = []
    for p in unclassified_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in ALL_VALID_EXTENSIONS:
            continue

        name_lower = p.name.lower()

        # Nếu file đã đổi tên thành người đã biết (ví dụ: binhan_..., low_binhan_...) -> BỎ QUA
        is_known_classified = False
        if name_lower.startswith("low_"):
            is_known_classified = True
        else:
            for p_name in known_person_names:
                if name_lower.startswith(f"{p_name}_"):
                    is_known_classified = True
                    break

        if is_known_classified:
            continue

        files.append(p)

    if not files:
        click.echo(f"⚠️  Không có file UNKNOWN nào cần gom nhóm trong '{unclassified_name}/'.")
        click.echo("💡 Lưu ý: Các file đã nhận diện (như binhan_..., lydao_...) không được gom nhóm.")
        return

    click.echo("=" * 60)
    click.echo("  Auto-Clustering Unknown Faces (Gom nhóm người lạ)")
    click.echo("=" * 60)
    click.echo(f"📌 Mode          : {'APPLY ✏️ (Tạo folder người mới & move)' if apply else 'DRY-RUN 🔍 (Chỉ xem trước)'}")
    click.echo(f"📌 Nguồn         : {unclassified_dir}")
    click.echo(f"📌 DBSCAN eps    : {dbscan_eps}")
    click.echo(f"📌 Min samples   : {dbscan_min_samples}")
    click.echo("")

    click.echo(f"🔍 Đang phân tích khuôn mặt {len(files)} file chưa phân loại...")

    encodings_list = []
    file_paths = []

    for fpath in tqdm(files, desc="Encoding", unit="file"):
        enc = _extract_encoding_from_file(fpath, model=model, sample_frames=sample_frames)
        if enc is not None:
            encodings_list.append(enc)
            file_paths.append(fpath)

    if not encodings_list:
        click.echo("⚠️  Không trích xuất được khuôn mặt nào từ các file trong chua_phan_loai.")
        return

    click.echo(f"✅ Trích xuất thành công {len(encodings_list)} khuôn mặt. Đang tiến hành gom nhóm (Clustering)...")

    # 2. Chạy DBSCAN clustering với metric Euclidean
    clt = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="euclidean")
    clt.fit(encodings_list)

    labels = clt.labels_
    label_set = set(labels)

    # Label -1 là noise (các mặt đơn lẻ không gom được thành nhóm)
    cluster_labels = [l for l in label_set if l != -1]

    if not cluster_labels:
        click.echo("\n⚠️  Không phát hiện nhóm người mới nào xuất hiện trùng lặp (tất cả là khuôn mặt đơn lẻ).")
        return

    click.echo(f"\n🎉 Phát hiện {len(cluster_labels)} nhóm người mới xuất hiện trùng lặp:\n")

    # Tìm index tiếp theo cho nguoi_moi_X
    existing_person_dirs = [d.name for d in dataset_root.iterdir() if d.is_dir()]
    next_idx = 1
    while f"{prefix_folder}_{next_idx}" in existing_person_dirs:
        next_idx += 1

    moved_total = 0

    for cluster_id in sorted(cluster_labels):
        cluster_files = [file_paths[i] for i, l in enumerate(labels) if l == cluster_id]
        folder_name = f"{prefix_folder}_{next_idx}"
        target_dir = dataset_root / folder_name

        click.echo(f"👤 [Nhóm {cluster_id + 1}] Gợi ý tạo folder '{folder_name}' ({len(cluster_files)} file):")
        for fpath in cluster_files:
            click.echo(f"   • {fpath.name}")

        if apply:
            target_dir.mkdir(exist_ok=True, parents=True)
            for fpath in cluster_files:
                # Làm sạch tên file (bỏ prefix unknown_ nếu có)
                clean_name = fpath.name
                if clean_name.lower().startswith(f"{unknown_prefix}_"):
                    clean_name = clean_name[len(unknown_prefix) + 1:]

                dest_file = target_dir / f"{folder_name}_{clean_name}"
                if dest_file.exists():
                    dest_file = target_dir / f"{folder_name}_{fpath.name}"

                shutil.move(fpath, dest_file)
                moved_total += 1

            click.echo(f"   ✅ Đã tạo '{folder_name}' & di chuyển {len(cluster_files)} file vào dataset!\n")

        next_idx += 1

    # 3. Xóa cache nếu apply để lần chạy sau tự động học người mới
    if apply and moved_total > 0:
        cache_file = dataset_root / cfg.get("cache_file", ".face_encodings_cache.pkl")
        if cache_file.exists():
            cache_file.unlink()
            click.echo("🔄 Đã tự động làm mới cache để cập nhật các người mới vào hệ thống.")

    click.echo("=" * 60)
    click.echo("  KẾT QUẢ CLUSTERING")
    click.echo("=" * 60)
    click.echo(f"  👥 Nhóm người mới tìm thấy : {len(cluster_labels)}")
    click.echo(f"  📁 File đã gom nhóm       : {sum(1 for l in labels if l != -1)}")
    click.echo(f"  ❓ File đơn lẻ (noise)     : {sum(1 for l in labels if l == -1)}")
    if not apply:
        click.echo("\n  💡 Chạy với '--apply' để tự động tạo folder người mới và chuyển file!")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
