#!/usr/bin/env python3
"""
Prepare Test Data Script
========================
Reset sạch thư mục `dataset/` từ `dataset-backup/`, dọn dẹp `chua_phan_loai/`,
sau đó di chuyển (move) ngẫu nhiên N ảnh/video từ tất cả folder người vào `chua_phan_loai/`
với tên file ẩn danh (IMG_0001.jpg...) để phục vụ test.

Cách dùng:
  python prepare_test_data.py               # Reset + di chuyển ngẫu nhiên 2 ảnh/người
  python prepare_test_data.py --count 3    # Reset + di chuyển ngẫu nhiên 3 ảnh/người
"""

import sys
import random
import shutil
from pathlib import Path
import click
import yaml

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff",
    ".mp4", ".avi", ".mov", ".mkv", ".webm"
}


@click.command()
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn file config YAML.")
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Thư mục đích để chuyển ảnh test (vd: 'other', 'chua_phan_loai').")
@click.option("--count", "-n", default=2, show_default=True, help="Số ảnh/video lấy từ mỗi thư mục người.")
@click.option("--copy", is_flag=True, default=False, help="Copy file thay vì di chuyển (move).")
def main(config: str, target_dir: str, count: int, copy: bool):
    cfg_path = Path(config)
    if not cfg_path.exists():
        click.echo(f"❌ Không tìm thấy file config: {config}", err=True)
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("dataset_root", "./dataset")).expanduser().resolve()
    unclassified_name = cfg.get("unclassified_dir", "chua_phan_loai")

    # 1. Kiểm tra backup dir
    backup_dir = dataset_root.parent / "dataset-backup"
    if not backup_dir.exists():
        backup_dir = dataset_root.parent / "dataset_backup"

    if backup_dir.exists():
        click.echo(f"🗑️  Xóa và khôi phục mới `dataset/` từ backup '{backup_dir.name}'...")
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        shutil.copytree(backup_dir, dataset_root)
        click.echo("✅ Đã khôi phục dataset sạch hoàn toàn!\n")
    else:
        click.echo(f"⚠️  Không thấy 'dataset_backup', giữ nguyên '{dataset_root.name}/'.")

    # 2. Đảm bảo thư mục chua_phan_loai tồn tại và sạch sẽ
    unclassified_dir = dataset_root / unclassified_name
    if unclassified_dir.exists():
        shutil.rmtree(unclassified_dir)
    unclassified_dir.mkdir(exist_ok=True, parents=True)

    # 3. Lấy danh sách thư mục người
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
    click.echo(f"📦 {action_name} ngẫu nhiên {count} ảnh/video từ mỗi folder sang '{unclassified_name}/'...\n")

    counter = 1
    total_moved = 0

    for person_dir in sorted(person_dirs):
        media_files = [p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
        if not media_files:
            click.echo(f"  ⚠️  Folder '{person_dir.name}' không có file media nào, bỏ qua.")
            continue

        selected = random.sample(media_files, min(count, len(media_files)))

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

    click.echo(f"\n✅ Hoàn tất! Đã {'copy' if copy else 'di chuyển'} tổng cộng {total_moved} file vào '{unclassified_name}/'.")


if __name__ == "__main__":
    main()
