#!/usr/bin/env python3
"""
Face Downloader CLI Tool
========================
Tự động tìm kiếm và tải ảnh khuôn mặt chất lượng cao từ web để bổ sung vào
tập dữ liệu huấn luyện (Training dataset) hoặc tập kiểm thử (Test data chua_phan_loai).

Cách dùng:
  # Tải 10 ảnh Sơn Tùng vào dataset/sontung
  python download_faces.py --query "Sơn Tùng M-TP" --person sontung --count 10

  # Tải 5 ảnh người lạ (ví dụ: Taylor Swift) vào chua_phan_loai để test gom nhóm DBSCAN
  python download_faces.py --query "Taylor Swift" --test --count 5
"""

import sys
from pathlib import Path
from typing import Optional
import click

from facecls.config import load_config
from facecls.engine import FaceEngine
from facecls.downloader import download_faces


@click.command()
@click.option("--query", "-q", required=True, help="Từ khóa tìm kiếm (tên người, hoặc @username Instagram).")
@click.option("--person", "-p", default=None, help="Tên thư mục người trong dataset (vd: sontung, trump).")
@click.option("--dest", "-d", default=None, help="Đường dẫn thư mục lưu ảnh tùy biến (vd: download/taylor_swift, download, chua_phan_loai).")
@click.option("--count", "-n", default=10, show_default=True, help="Số lượng ảnh cần tải.")
@click.option("--test", is_flag=True, default=False, help="Lưu trực tiếp vào thư mục chua_phan_loai làm ảnh test ẩn danh.")
@click.option("--instagram", "--ig", is_flag=True, default=False, help="Cào trực tiếp ảnh từ tài khoản Instagram công khai (@username).")
@click.option("--facebook", "--fb", is_flag=True, default=False, help="Cào trực tiếp ảnh từ Facebook profile hoặc album ảnh.")
@click.option("--verify/--no-verify", default=True, show_default=True, help="Kiểm tra đảm bảo ảnh có khuôn mặt rõ nét.")
@click.option("--same-person/--no-same-person", default=True, show_default=True, help="Tự động trích xuất vector mỏ neo để đảm bảo các ảnh sau cùng 1 người.")
@click.option("--threshold", default=0.38, show_default=True, type=float, help="Ngưỡng Cosine Similarity để công nhận cùng 1 người.")
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn tệp cấu hình.")
def main(
    query: str,
    person: Optional[str],
    dest: Optional[str],
    count: int,
    test: bool,
    instagram: bool,
    facebook: bool,
    verify: bool,
    same_person: bool,
    threshold: float,
    config: str,
):
    """Công cụ tìm kiếm và tải ảnh khuôn mặt từ web, Instagram và Facebook."""
    try:
        cfg = load_config(config)
    except Exception:
        cfg = None

    import re
    from facecls.downloader import resolve_facebook_profile_info, resolve_instagram_profile_info
    project_root = Path(__file__).resolve().parent
    dataset_root = Path(cfg.dataset_root if cfg else "./dataset").expanduser().resolve()
    
    if "facebook.com" in query or "fb.com" in query:
        clean_handle, _, _, _ = resolve_facebook_profile_info(query)
        clean_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", clean_handle.lower()).strip("_")
    elif "instagram.com" in query or query.startswith("@"):
        clean_handle, _, _ = resolve_instagram_profile_info(query)
        clean_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", clean_handle.lower()).strip("_")
    else:
        clean_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", query.strip().lstrip("@").lower()).strip("_")

    if test:
        unclassified_name = cfg.unclassified_dir if cfg else "chua_phan_loai"
        output_dir = dataset_root / unclassified_name
        prefix = "IMG_TEST"
    elif dest:
        dest_path = Path(dest).expanduser()
        if not dest_path.is_absolute():
            dest_path = (project_root / dest_path).resolve()
        
        # Nếu truyền 'download' hoặc 'download/' -> Tự động lưu vào download/<tên_người>
        if dest_path.name.lower() == "download" or str(dest_path).endswith("/download"):
            output_dir = dest_path / clean_slug
        else:
            output_dir = dest_path

        prefix = person.lower().strip() if person else clean_slug
    elif person:
        output_dir = dataset_root / person.lower().strip()
        prefix = person.lower().strip()
    else:
        # Mặc định lưu vào thư mục download/<tên_người> thay vì xáo trộn dataset
        output_dir = project_root / "download" / clean_slug
        prefix = clean_slug

    click.echo("=" * 60)
    click.echo("  Face Downloader Tool")
    click.echo("=" * 60)
    click.echo(f"🔍 Từ khóa       : {query}")
    click.echo(f"📁 Thư mục lưu   : {output_dir}")
    click.echo(f"🎯 Số lượng cần  : {count} ảnh")
    click.echo(f"🛡️ Kiểm duyệt mặt: {'Bật' if verify else 'Tắt'}")
    click.echo("")

    engine_obj = None
    if verify and cfg:
        click.echo("🚀 Đang khởi động FaceEngine để kiểm tra chất lượng khuôn mặt...")
        engine_obj = FaceEngine(
            engine=cfg.engine,
            insightface_model=cfg.insightface_model,
            allowed_modules=cfg.allowed_modules,
            det_thresh=cfg.det_thresh,
            multi_face_strategy=cfg.multi_face_strategy,
            image_max_size=cfg.image_max_size,
        )

    saved = download_faces(
        query=query,
        output_dir=output_dir,
        target_count=count,
        prefix=prefix,
        verify_face=verify,
        same_person=same_person,
        similarity_threshold=threshold,
        engine_obj=engine_obj,
        is_instagram=instagram,
        is_facebook=facebook,
    )

    click.echo("\n" + "=" * 60)
    click.echo(f"✅ Hoàn tất! Đã tải thành công {saved}/{count} ảnh vào: {output_dir}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
