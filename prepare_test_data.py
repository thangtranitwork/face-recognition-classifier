#!/usr/bin/env python3
"""
Prepare Test Data Script
========================
Di chuyển (hoặc copy) ngẫu nhiên N ảnh từ các thư mục người trong `dataset/`
sang thư mục `chua_phan_loai/` và đổi tên thành dạng IMG_0001.jpg
để phục vụ kiểm thử tool.

Cách dùng:
  python prepare_test_data.py               # Di chuyển ngẫu nhiên 2 ảnh/người
  python prepare_test_data.py --count 3    # Di chuyển ngẫu nhiên 3 ảnh/người
  python prepare_test_data.py --copy       # Copy thay vì di chuyển (move)
"""

import sys
import random
import shutil
from pathlib import Path
import click
import yaml

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


@click.command()
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn file config YAML.")
@click.option("--count", "-n", default=2, show_default=True, help="Số ảnh lấy từ mỗi thư mục người.")
@click.option("--copy", is_flag=True, default=False, help="Copy ảnh thay vì di chuyển (move).")
def main(config: str, count: int, copy: bool):
    cfg_path = Path(config)
    if not cfg_path.exists():
        click.echo(f"❌ Không tìm thấy file config: {config}", err=True)
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("dataset_root", "./dataset")).expanduser().resolve()
    unclassified_name = cfg.get("unclassified_dir", "chua_phan_loai")
    unclassified_dir = dataset_root / unclassified_name
    unclassified_dir.mkdir(exist_ok=True, parents=True)

    exclude_dirs = {d.lower() for d in cfg.get("exclude_dirs", [])}
    exclude_dirs.add(unclassified_name.lower())

    person_dirs = [
        d for d in dataset_root.iterdir()
        if d.is_dir() and d.name.lower() not in exclude_dirs
    ]

    if not person_dirs:
        click.echo("❌ Không tìm thấy thư mục người nào trong dataset.")
        sys.exit(1)

    action_name = "Copying" if copy else "Moving"
    click.echo(f"📦 {action_name} {count} ảnh từ mỗi folder sang '{unclassified_name}/'...\n")

    # Tìm index tiếp theo cho tên file IMG_XXXX
    existing_imgs = list(unclassified_dir.glob("IMG_*"))
    max_idx = 0
    for img in existing_imgs:
        stem = img.stem.replace("IMG_", "")
        if stem.isdigit():
            max_idx = max(max_idx, int(stem))

    counter = max_idx + 1
    total_moved = 0

    for person_dir in sorted(person_dirs):
        images = [p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
        if not images:
            click.echo(f"  ⚠️  Folder '{person_dir.name}' không có ảnh nào, bỏ qua.")
            continue

        selected = random.sample(images, min(count, len(images)))

        for src in selected:
            dest_name = f"IMG_{counter:04d}{src.suffix.lower()}"
            dest_path = unclassified_dir / dest_name

            if copy:
                shutil.copy2(src, dest_path)
            else:
                shutil.move(src, dest_path)

            click.echo(f"  {'📋' if copy else '🚚'} {person_dir.name}/{src.name} → {unclassified_name}/{dest_name}")
            counter += 1
            total_moved += 1

    click.echo(f"\n✅ Hoàn tất! Đã {'copy' if copy else 'di chuyển'} tổng cộng {total_moved} ảnh vào '{unclassified_name}/'.")


if __name__ == "__main__":
    main()
