#!/usr/bin/env python3
"""
Face Recognition File Renamer
==============================
Tự động nhận diện khuôn mặt trong ảnh/video chưa phân loại và đổi tên tệp
theo danh tính người được nhận diện bằng mô hình học sâu InsightFace (ArcFace).
"""

import sys
from pathlib import Path
from typing import Optional
import click

from facecls.config import load_config
from facecls.cache import EmbeddingCache
from facecls.engine import get_insightface_app, FaceEngine
from facecls.pipeline import run_rename, load_known_faces

__all__ = ["get_insightface_app", "FaceEngine", "load_known_faces", "run_rename"]


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
    "--unclassified-dir", "-u", "-d", "--dir", "--target-dir", "target_dir",
    default=None,
    help="Tên hoặc đường dẫn thư mục cần xử lý (vd: 'other', 'chua_phan_loai').",
)
def main(config: str, apply: bool, clear_cache: bool, target_dir: Optional[str]):
    """
    \b
    Face Recognition File Renamer
    ─────────────────────────────
    Nhận diện khuôn mặt và đổi tên ảnh/video trong thư mục chưa phân loại.
    """
    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)

    dataset_root = Path(cfg.dataset_root).expanduser().resolve()
    cache_path = dataset_root / cfg.cache_file
    cache = EmbeddingCache(cache_path)

    if clear_cache:
        cache.clear()
        click.echo(f"🗑️ Đã xóa cache: {cache_path}")

    run_rename(cfg=cfg, cache=cache, target_dir=target_dir, apply=apply)


if __name__ == "__main__":
    main()
