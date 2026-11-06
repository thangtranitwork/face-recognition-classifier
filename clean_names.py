#!/usr/bin/env python3
"""
Clean / Normalize Dataset File Names Script
===========================================
Đổi tên tất cả các file ảnh/video trong các thư mục người thuộc dataset
thành dạng chuẩn: <tên_thư_mục>_<stt>.<extension>
"""

import sys
from pathlib import Path
from typing import Optional
import click

from facecls.config import load_config
from facecls.pipeline import run_clean


@click.command()
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn file config YAML.")
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Chỉ định 1 thư mục người cụ thể (vd: binhan, sontung).")
@click.option("--apply", "-a", is_flag=True, default=False, help="Thực sự đổi tên file (mặc định là dry-run).")
def main(config: str, target_dir: Optional[str], apply: bool):
    """
    \b
    Normalize Dataset File Names Tool
    ─────────────────────────────────
    Chuẩn hóa tên tệp trong thư mục người thành <tên>_<STT>.<ext>.
    """
    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)

    run_clean(cfg=cfg, target_dir=target_dir, apply=apply)


if __name__ == "__main__":
    main()
