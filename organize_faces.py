#!/usr/bin/env python3
"""
Organize Renamed Faces Script
=============================
Tự động quét các ảnh đã đổi tên trong thư mục `chua_phan_loai/`
và di chuyển (move) chúng vào đúng thư mục người tương ứng.

Đặc biệt:
- BỎ QUA các file `unknown_*` (giữ nguyên trong chua_phan_loai).
- Xử lý các file dạng `person_A_IMG_001.jpg` hoặc `low_person_A_IMG_001.jpg`.

Cách dùng:
  python organize_faces.py             # Dry-run: xem trước các file sẽ di chuyển
  python organize_faces.py --apply     # Thực sự di chuyển file
"""

import sys
import shutil
from pathlib import Path
import click
import yaml

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


@click.command()
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn file config YAML.")
@click.option("--apply", is_flag=True, default=False, help="Thực sự di chuyển file (mặc định là dry-run).")
@click.option("--include-low", is_flag=True, default=True, help="Di chuyển cả ảnh low_confidence (mặc định: True).")
def main(config: str, apply: bool, include_low: bool):
    cfg_path = Path(config)
    if not cfg_path.exists():
        click.echo(f"❌ Không tìm thấy file config: {config}", err=True)
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("dataset_root", "./dataset")).expanduser().resolve()
    unclassified_name = cfg.get("unclassified_dir", "chua_phan_loai")
    unclassified_dir = dataset_root / unclassified_name
    unknown_prefix = cfg.get("unknown_prefix", "unknown").lower()

    if not unclassified_dir.exists():
        click.echo(f"❌ Thư mục '{unclassified_dir}' không tồn tại.", err=True)
        sys.exit(1)

    # Thu thập danh sách thư mục người hợp lệ
    exclude_dirs = {d.lower() for d in cfg.get("exclude_dirs", [])}
    exclude_dirs.add(unclassified_name.lower())

    person_dirs = {
        d.name.lower(): d for d in dataset_root.iterdir()
        if d.is_dir() and d.name.lower() not in exclude_dirs
    }

    images = [
        p for p in unclassified_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    if not images:
        click.echo(f"⚠️  Không có ảnh nào trong '{unclassified_name}/'")
        return

    click.echo("=" * 60)
    click.echo("  Organize Renamed Faces Tool")
    click.echo("=" * 60)
    click.echo(f"📌 Mode: {'APPLY ✏️ (Di chuyển thật)' if apply else 'DRY-RUN 🔍 (Chỉ xem trước)'}")
    click.echo(f"📌 Nguồn: {unclassified_dir}")
    click.echo(f"📌 Đã loại trừ: {unknown_prefix}_* (không di chuyển)")
    click.echo("")

    moved_count = 0
    skipped_unknown = 0
    unmatched_count = 0

    for img in sorted(images):
        name_lower = img.name.lower()

        # 1. Bỏ qua file unknown (theo yêu cầu người dùng)
        if name_lower.startswith(f"{unknown_prefix}_"):
            click.echo(f"  ⏭️  Bỏ qua (unknown): {img.name}")
            skipped_unknown += 1
            continue

        # 2. Xử lý file low confidence nếu có
        clean_name = img.name
        is_low = False
        if name_lower.startswith("low_"):
            if not include_low:
                click.echo(f"  ⏭️  Bỏ qua (low_confidence): {img.name}")
                continue
            clean_name = img.name[4:]  # Bỏ 'low_'
            is_low = True

        # 3. Khớp tên file với tên thư mục người
        matched_person = None
        matched_dir = None

        # Sắp xếp tên folder dài trước để khớp chính xác
        for p_name, p_dir in sorted(person_dirs.items(), key=lambda x: len(x[0]), reverse=True):
            if clean_name.lower().startswith(f"{p_name}_"):
                matched_person = p_dir.name
                matched_dir = p_dir
                break

        if not matched_dir:
            click.echo(f"  ⚠️  Không khớp thư mục nào: {img.name}")
            unmatched_count += 1
            continue

        dest_file = matched_dir / img.name

        # Xử lý trùng tên file nếu có
        if dest_file.exists():
            counter = 1
            stem = img.stem
            suffix = img.suffix
            while dest_file.exists():
                dest_file = matched_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        prefix_tag = "⚠️  LOW" if is_low else "✅ HIGH"
        click.echo(f"  🚚 [{prefix_tag}] {img.name} → {matched_person}/{dest_file.name}")
        moved_count += 1

        if apply:
            try:
                shutil.move(img, dest_file)
            except Exception as e:
                click.echo(f"  ❌ Lỗi di chuyển {img.name}: {e}", err=True)

    click.echo("\n" + "=" * 60)
    click.echo("  KẾT QUẢ")
    click.echo("=" * 60)
    click.echo(f"  🚚 Đã di chuyển    : {moved_count}")
    click.echo(f"  ⏭️  Bỏ qua unknown : {skipped_unknown}")
    click.echo(f"  ⚠️  Không khớp     : {unmatched_count}")
    if not apply:
        click.echo("\n  💡 Chạy với --apply để thực sự di chuyển file.")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
