#!/usr/bin/env python3
"""
Clean / Normalize Dataset File Names Script
===========================================
Đổi tên tất cả các file ảnh/video trong các thư mục người thuộc dataset
thành dạng chuẩn: <tên_thư_mục>_<stt>.<extension>

Ví dụ:
  dataset/binhan/images (3).jpeg → dataset/binhan/binhan_1.jpeg
  dataset/binhan/471741752_n.jpg → dataset/binhan/binhan_2.jpg

Cách dùng:
  python clean_names.py              # Dry-run xem trước
  python clean_names.py --apply      # Thực sự đổi tên
  python clean_names.py -d binhan    # Chỉ đổi tên trong thư mục binhan
"""

import sys
from pathlib import Path
from typing import Optional
import click
import yaml

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff",
    ".mp4", ".avi", ".mov", ".mkv", ".webm"
}


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
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Chỉ định 1 thư mục người cụ thể (vd: binhan, sontung).")
@click.option("--apply", "-a", is_flag=True, default=False, help="Thực sự đổi tên file (mặc định là dry-run).")
def main(config: str, target_dir: Optional[str], apply: bool):
    cfg_path = Path(config)
    if not cfg_path.exists():
        click.echo(f"❌ Không tìm thấy file config: {config}", err=True)
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("dataset_root", "./dataset")).expanduser().resolve()
    unclassified_name = cfg.get("unclassified_dir", "chua_phan_loai")
    exclude_dirs = {d.lower() for d in cfg.get("exclude_dirs", [])}
    exclude_dirs.add(unclassified_name.lower())

    if target_dir:
        # Nếu truyền cụ thể 1 folder người (vd: -d binhan)
        p = resolve_target_dir(target_dir, unclassified_name, dataset_root)
        if p.name.lower() in exclude_dirs:
            click.echo(f"❌ Thư mục '{p.name}' nằm trong danh sách loại trừ (exclude_dirs).", err=True)
            sys.exit(1)
        person_dirs = [p]
    else:
        # Tất cả các thư mục người trong dataset
        person_dirs = [
            d for d in dataset_root.iterdir()
            if d.is_dir() and d.name.lower() not in exclude_dirs
        ]

    if not person_dirs:
        click.echo("⚠️  Không tìm thấy thư mục người nào để chuẩn hóa tên file.")
        return

    click.echo("=" * 60)
    click.echo("  Normalize Dataset File Names Tool (Tên + STT)")
    click.echo("=" * 60)
    click.echo(f"📌 Mode: {'APPLY ✏️ (Đổi tên thật)' if apply else 'DRY-RUN 🔍 (Chỉ xem trước)'}")
    click.echo(f"📌 Số thư mục xử lý: {len(person_dirs)}")
    click.echo("")

    total_renamed = 0

    for p_dir in sorted(person_dirs, key=lambda d: d.name.lower()):
        person_name = p_dir.name
        files = sorted([
            f for f in p_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ], key=lambda f: f.name.lower())

        if not files:
            click.echo(f"📁 [{person_name}] Không có file media nào.")
            continue

        click.echo(f"📁 [{person_name}] {len(files)} file:")

        # 1. Tạo danh sách mapping tên cũ -> tên mới chuẩn: <person_name>_<stt><ext>
        rename_pairs = []
        for idx, fpath in enumerate(files, start=1):
            new_name = f"{person_name}_{idx}{fpath.suffix.lower()}"
            rename_pairs.append((fpath, new_name))

        for fpath, new_name in rename_pairs:
            click.echo(f"   • {fpath.name} → {new_name}")

        if apply:
            # Bước A: Đổi sang tên tạm thời trùng tránh ghi đè
            temp_pairs = []
            for idx, (fpath, new_name) in enumerate(rename_pairs, start=1):
                temp_path = fpath.parent / f"__temp_norm_{idx}__{fpath.name}"
                try:
                    fpath.rename(temp_path)
                    temp_pairs.append((temp_path, new_name))
                except Exception as e:
                    click.echo(f"   ❌ Lỗi đổi tên tạm {fpath.name}: {e}", err=True)

            # Bước B: Đổi từ tên tạm sang tên chuẩn final
            for temp_path, new_name in temp_pairs:
                final_path = temp_path.parent / new_name
                try:
                    temp_path.rename(final_path)
                    total_renamed += 1
                except Exception as e:
                    click.echo(f"   ❌ Lỗi đổi tên chính thức {temp_path.name}: {e}", err=True)

        else:
            total_renamed += len(rename_pairs)

        click.echo("")

    # Xóa cache nếu apply để lần sau build lại encodings mới
    if apply and total_renamed > 0:
        cache_file = dataset_root / cfg.get("cache_file", ".face_encodings_cache.pkl")
        if cache_file.exists():
            cache_file.unlink()
            click.echo("🔄 Đã tự động xóa cache để build lại fingerprint mới.")

    click.echo("=" * 60)
    click.echo("  KẾT QUẢ")
    click.echo("=" * 60)
    click.echo(f"  ✏️  Tổng số file {'đã đổi tên' if apply else 'sẽ đổi tên'}: {total_renamed}")
    if not apply:
        click.echo("\n  💡 Chạy với --apply (hoặc -a) để thực sự đổi tên file trong các folder.")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
