#!/usr/bin/env python3
"""
Prepare Test Data Script
========================
Reset sạch thư mục dataset từ dataset_backup (nếu có),
sao chép an toàn N ảnh/video từ tất cả folder người vào chua_phan_loai
với tên file ẩn danh (IMG_0001.jpg...) để phục vụ kiểm thử.
"""

import sys
from pathlib import Path
from typing import Optional
import click

from facecls.config import load_config
from facecls.pipeline import run_prepare


@click.command()
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn file config YAML.")
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Thư mục đích để chuyển ảnh test.")
@click.option("--count", "-n", default=3, show_default=True, help="Số ảnh/video lấy từ mỗi thư mục người.")
@click.option("--copy/--move", default=True, show_default=True, help="Copy file (an toàn) thay vì di chuyển (move).")
@click.option("--include-other/--no-other", default=True, show_default=True, help="Bao gồm cả ảnh người lạ từ 'other/' để kiểm thử gom nhóm.")
def main(config: str, target_dir: Optional[str], count: int, copy: bool, include_other: bool):
    """
    \b
    Prepare Test Data Tool
    ──────────────────────
    Khôi phục từ backup và chuẩn bị dữ liệu kiểm thử an toàn.
    """
    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)

    run_prepare(cfg=cfg, target_dir=target_dir, count=count, copy=copy, include_other=include_other)


if __name__ == "__main__":
    main()
