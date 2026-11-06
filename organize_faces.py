#!/usr/bin/env python3
"""
Organize Renamed Faces Script
=============================
Tự động quét các ảnh đã đổi tên trong thư mục chưa phân loại và di chuyển (move)
chúng vào đúng thư mục người tương ứng trong dataset.

Đặc biệt:
- BỎ QUA các file unknown_* (giữ nguyên).
- Phát hiện các nhóm người mới (dạng nguoi_moi_1_...) và hỏi tên thật trước khi tạo folder.
"""

import sys
from pathlib import Path
from typing import Optional
import click

from facecls.config import load_config
from facecls.pipeline import run_organize


@click.command()
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn file config YAML.")
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Thư mục chứa file đã đổi tên (vd: 'other', 'chua_phan_loai').")
@click.option("--apply", "-a", is_flag=True, default=False, help="Thực sự di chuyển file (mặc định là dry-run).")
@click.option("--yes", "-y", is_flag=True, default=False, help="Tự động lấy tên mặc định cho người mới mà không hỏi.")
@click.option("--include-low/--no-include-low", default=True, show_default=True, help="Di chuyển cả ảnh low_confidence.")
def main(config: str, target_dir: Optional[str], apply: bool, yes: bool, include_low: bool):
    """
    \b
    Organize Renamed Faces Tool
    ───────────────────────────
    Di chuyển tệp đã đổi tên về thư mục người trong dataset.
    """
    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)

    run_organize(cfg=cfg, target_dir=target_dir, apply=apply, yes=yes, include_low=include_low)


if __name__ == "__main__":
    main()
