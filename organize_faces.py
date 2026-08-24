#!/usr/bin/env python3
"""
Organize Renamed Faces Script
=============================
Tự động quét các ảnh đã đổi tên trong thư mục `chua_phan_loai/`
và di chuyển (move) chúng vào đúng thư mục người tương ứng.

Đặc biệt:
- BỎ QUA các file `unknown_*` (giữ nguyên trong chua_phan_loai).
- Phát hiện các nhóm người mới (dạng `nguoi_moi_1_...`) và HỎI TÊN THẬT của người mới
  trước khi tạo folder trong dataset & di chuyển file!
"""

import sys
import shutil
import re
from pathlib import Path
from typing import Optional, Dict, List
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


INVALID_PREFIXES = {"img", "image", "images", "photo", "screenshot", "screencast", "unknown", "low", "video", "media", "frame"}


def extract_candidate_prefix(filename: str, unknown_prefix: str) -> Optional[str]:
    name_lower = filename.lower()
    if name_lower.startswith("low_"):
        name_lower = name_lower[4:]

    if name_lower.startswith(f"{unknown_prefix}_"):
        return None

    # Khớp nguoi_moi_1, nguoi_moi_2...
    m = re.match(r"^(nguoi_moi_\d+)_", name_lower)
    if m:
        return m.group(1)

    # Tách prefix trước dấu _ cuối cùng hoặc _IMG_
    parts = name_lower.split("_")
    if len(parts) >= 2:
        if parts[0] == "nguoi" and len(parts) >= 3 and parts[1] == "moi":
            return f"{parts[0]}_{parts[1]}_{parts[2]}"
        candidate = parts[0].strip().lower()
        if candidate not in INVALID_PREFIXES and not candidate.isdigit():
            return candidate

    return None


@click.command()
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn file config YAML.")
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Thư mục chứa file đã đổi tên (vd: 'other', 'chua_phan_loai').")
@click.option("--apply", "-a", is_flag=True, default=False, help="Thực sự di chuyển file (mặc định là dry-run).")
@click.option("--yes", "-y", is_flag=True, default=False, help="Tự động lấy tên mặc định cho người mới mà không hỏi.")
@click.option("--include-low", is_flag=True, default=True, help="Di chuyển cả ảnh low_confidence (mặc định: True).")
def main(config: str, target_dir: Optional[str], apply: bool, yes: bool, include_low: bool):
    cfg_path = Path(config)
    if not cfg_path.exists():
        click.echo(f"❌ Không tìm thấy file config: {config}", err=True)
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("dataset_root", "./dataset")).expanduser().resolve()
    unclassified_name = cfg.get("unclassified_dir", "chua_phan_loai")
    unclassified_dir = resolve_target_dir(target_dir, unclassified_name, dataset_root)
    unknown_prefix = cfg.get("unknown_prefix", "unknown").lower()

    # Thu thập danh sách thư mục người hợp lệ
    exclude_dirs = {d.lower() for d in cfg.get("exclude_dirs", [])}
    exclude_dirs.add(unclassified_dir.name.lower())

    person_dirs: Dict[str, Path] = {
        d.name.lower(): d for d in dataset_root.iterdir()
        if d.is_dir() and d.name.lower() not in exclude_dirs
    }

    images = [
        p for p in unclassified_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    if not images:
        click.echo(f"⚠️  Không có file nào trong '{unclassified_dir.name}/'")
        return

    click.echo("=" * 60)
    click.echo("  Organize Renamed Faces Tool")
    click.echo("=" * 60)
    click.echo(f"📌 Mode: {'APPLY ✏️ (Di chuyển thật)' if apply else 'DRY-RUN 🔍 (Chỉ xem trước)'}")
    click.echo(f"📌 Nguồn: {unclassified_dir}")
    click.echo(f"📌 Đã loại trừ: {unknown_prefix}_* (không di chuyển)")
    click.echo("")

    # 1. Tìm các file chưa khớp thư mục người nào -> Nhóm theo prefix người mới
    unmatched_files = []
    for img in sorted(images):
        name_lower = img.name.lower()
        if name_lower.startswith(f"{unknown_prefix}_"):
            continue

        clean_name = img.name[4:] if name_lower.startswith("low_") else img.name

        matched = False
        for p_name in sorted(person_dirs.keys(), key=len, reverse=True):
            if clean_name.lower().startswith(f"{p_name}_"):
                matched = True
                break

        if not matched:
            unmatched_files.append(img)

    # Gom nhóm unmatched_files theo prefix người mới
    new_people_groups: Dict[str, List[Path]] = {}
    for img in unmatched_files:
        prefix = extract_candidate_prefix(img.name, unknown_prefix)
        if prefix:
            new_people_groups.setdefault(prefix, []).append(img)

    created_new_dirs = False

    # 2. Xử lý hỏi tên người mới nếu có nhóm chưa đăng ký
    if new_people_groups:
        for old_prefix, group_files in sorted(new_people_groups.items()):
            if old_prefix.lower() in person_dirs:
                continue

            click.echo("─" * 60)
            click.echo(f"👤 Phát hiện nhóm người mới '{old_prefix}' ({len(group_files)} file):")
            for f in group_files:
                click.echo(f"   • {f.name}")

            if apply:
                if yes or not sys.stdin.isatty():
                    chosen_name = old_prefix
                else:
                    chosen_name = click.prompt(
                        f"👉 Nhập tên người thật cho nhóm '{old_prefix}' (Enter để giữ nguyên)",
                        default=old_prefix,
                        show_default=True
                    ).strip().lower()

                if not chosen_name:
                    chosen_name = old_prefix

                new_dir = dataset_root / chosen_name
                new_dir.mkdir(exist_ok=True, parents=True)
                person_dirs[chosen_name.lower()] = new_dir
                created_new_dirs = True

                # Nếu người dùng đổi tên từ 'nguoi_moi_1' -> 'hoang_thuy_linh', tiến hành đổi tên file
                if chosen_name.lower() != old_prefix.lower():
                    updated_group_files = []
                    for f in group_files:
                        orig_name = f.name
                        # Thay thế prefix cũ bằng tên mới
                        if orig_name.lower().startswith("low_"):
                            new_fname = f"low_{chosen_name}_" + orig_name[4 + len(old_prefix) + 1:]
                        elif orig_name.lower().startswith(f"{old_prefix.lower()}_"):
                            new_fname = f"{chosen_name}_" + orig_name[len(old_prefix) + 1:]
                        else:
                            new_fname = f"{chosen_name}_{orig_name}"

                        new_fpath = f.parent / new_fname
                        f.rename(new_fpath)
                        updated_group_files.append(new_fpath)
                        click.echo(f"   ✏️  Đổi tên file: {orig_name} → {new_fname}")
                    # Cập nhật danh sách file sau khi đổi tên
                    images = [p for p in unclassified_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]

                click.echo(f"   ✅ Đã tạo thư mục 'dataset/{chosen_name}/'")
            else:
                person_dirs[old_prefix.lower()] = dataset_root / old_prefix
                click.echo(f"   💡 Chạy với -a để gán tên người thật và tự động tạo thư mục 'dataset/{old_prefix}/'")

    click.echo("\n" + "=" * 60)
    click.echo("  TIẾN HÀNH DI CHUYỂN FILE VÀO DATASET")
    click.echo("=" * 60)

    # 3. Tiến hành di chuyển các file đã khớp vào đúng folder người
    moved_count = 0
    skipped_unknown = 0
    unmatched_count = 0

    for img in sorted(images):
        name_lower = img.name.lower()

        # Bỏ qua unknown
        if name_lower.startswith(f"{unknown_prefix}_"):
            click.echo(f"  ⏭️  Bỏ qua (unknown): {img.name}")
            skipped_unknown += 1
            continue

        clean_name = img.name
        is_low = False
        if name_lower.startswith("low_"):
            if not include_low:
                click.echo(f"  ⏭️  Bỏ qua (low_confidence): {img.name}")
                continue
            clean_name = img.name[4:]
            is_low = True

        matched_person = None
        matched_dir = None

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

    # Nếu có tạo folder người mới thì làm mới cache encodings
    if apply and created_new_dirs:
        cache_file = dataset_root / cfg.get("cache_file", ".face_encodings_cache.pkl")
        if cache_file.exists():
            cache_file.unlink()
            click.echo("\n🔄 Đã tự động làm mới cache để cập nhật các người mới vào hệ thống.")

    click.echo("\n" + "=" * 60)
    click.echo("  KẾT QUẢ")
    click.echo("=" * 60)
    click.echo(f"  🚚 Đã di chuyển    : {moved_count}")
    click.echo(f"  ⏭️  Bỏ qua unknown : {skipped_unknown}")
    click.echo(f"  ⚠️  Không khớp     : {unmatched_count}")
    if not apply:
        click.echo("\n  💡 Chạy với --apply (hoặc -a) để thực sự di chuyển file.")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
