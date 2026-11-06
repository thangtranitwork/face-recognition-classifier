#!/usr/bin/env python3
"""
Auto-Clustering Unknown Faces Script
====================================
Tự động quét các ảnh/video chưa nhận diện trong thư mục chưa phân loại,
sử dụng thuật toán DBSCAN gom nhóm các khuôn mặt của cùng một người lạ xuất hiện nhiều lần.
"""

import sys
from pathlib import Path
from typing import Optional
import click

from facecls.config import load_config
from facecls.cache import EmbeddingCache
from facecls.pipeline import run_cluster


@click.command()
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn file config YAML.")
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Thư mục cần gom nhóm (vd: 'other', 'chua_phan_loai').")
@click.option("--apply", "-a", is_flag=True, default=False, help="Thực sự đổi tên file đã gom nhóm.")
@click.option("--eps", default=None, type=float, help="Override ngưỡng khoảng cách DBSCAN (default trong config).")
@click.option("--min-samples", default=None, type=int, help="Override số ảnh tối thiểu/nhóm (default trong config).")
@click.option("--clear-cache", "-cc", is_flag=True, default=False, help="Xóa cache encodings gom nhóm.")
@click.option("--workers", "-w", default=4, type=int, show_default=True, help="Số thread xử lý đọc ảnh song song.")
def main(
    config: str,
    target_dir: Optional[str],
    apply: bool,
    eps: Optional[float],
    min_samples: Optional[int],
    clear_cache: bool,
    workers: int
):
    """
    \b
    Auto-Clustering Unknown Faces Tool
    ──────────────────────────────────
    Gom nhóm tự động khuôn mặt người lạ bằng DBSCAN.
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

    run_cluster(
        cfg=cfg,
        cache=cache,
        target_dir=target_dir,
        apply=apply,
        override_eps=eps,
        override_min_samples=min_samples,
        workers=workers,
    )


if __name__ == "__main__":
    main()
