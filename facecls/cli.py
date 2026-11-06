"""Giao diện dòng lệnh hợp nhất (Unified CLI) cho hệ thống phân loại khuôn mặt."""

import sys
from pathlib import Path
from typing import Optional
import click

from facecls.config import load_config
from facecls.engine import FaceEngine
from facecls.cache import EmbeddingCache
from facecls.pipeline import (
    run_rename,
    run_cluster,
    run_organize,
    run_clean,
    run_prepare,
)


@click.group()
def cli():
    """Công cụ nhận diện, gom nhóm và tổ chức tập dữ liệu khuôn mặt."""
    pass


@cli.command("rename")
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn tệp config YAML.")
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Thư mục cần xử lý.")
@click.option("--apply", "-a", is_flag=True, default=False, help="Thực sự đổi tên tệp (mặc định Dry-run).")
@click.option("--clear-cache", "-cc", is_flag=True, default=False, help="Xóa bộ nhớ đệm cache.")
def cmd_rename(config: str, target_dir: Optional[str], apply: bool, clear_cache: bool):
    """Nhận diện khuôn mặt và đổi tên tệp."""
    cfg = load_config(config)
    dataset_root = Path(cfg.dataset_root).expanduser().resolve()
    cache_path = dataset_root / cfg.cache_file
    cache = EmbeddingCache(cache_path)

    if clear_cache:
        cache.clear()
        click.echo("🗑️ Đã xóa cache.")

    run_rename(cfg=cfg, cache=cache, target_dir=target_dir, apply=apply)


@cli.command("cluster")
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn tệp config YAML.")
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Thư mục cần gom nhóm.")
@click.option("--apply", "-a", is_flag=True, default=False, help="Thực sự đổi tên tệp đã gom nhóm.")
@click.option("--eps", default=None, type=float, help="Ngưỡng khoảng cách DBSCAN override.")
@click.option("--min-samples", default=None, type=int, help="Số ảnh tối thiểu mỗi nhóm override.")
@click.option("--clear-cache", "-cc", is_flag=True, default=False, help="Xóa cache.")
@click.option("--workers", "-w", default=4, type=int, show_default=True, help="Số luồng đọc ảnh.")
def cmd_cluster(config: str, target_dir: Optional[str], apply: bool, eps: Optional[float], min_samples: Optional[int], clear_cache: bool, workers: int):
    """Gom nhóm tự động khuôn mặt người lạ (DBSCAN)."""
    cfg = load_config(config)
    dataset_root = Path(cfg.dataset_root).expanduser().resolve()
    cache_path = dataset_root / cfg.cache_file
    cache = EmbeddingCache(cache_path)

    if clear_cache:
        cache.clear()
        click.echo("🗑️ Đã xóa cache.")

    run_cluster(
        cfg=cfg,
        cache=cache,
        target_dir=target_dir,
        apply=apply,
        override_eps=eps,
        override_min_samples=min_samples,
        workers=workers,
    )


@cli.command("organize")
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn tệp config YAML.")
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Thư mục chứa tệp đã đổi tên.")
@click.option("--apply", "-a", is_flag=True, default=False, help="Thực sự di chuyển tệp.")
@click.option("--yes", "-y", is_flag=True, default=False, help="Chấp nhận tên mặc định không cần hỏi.")
@click.option("--include-low/--no-include-low", default=True, help="Bao gồm hoặc bỏ qua tệp low_confidence.")
def cmd_organize(config: str, target_dir: Optional[str], apply: bool, yes: bool, include_low: bool):
    """Di chuyển các tệp đã đổi tên vào đúng thư mục người tương ứng."""
    cfg = load_config(config)
    run_organize(cfg=cfg, target_dir=target_dir, apply=apply, yes=yes, include_low=include_low)


@cli.command("clean")
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn tệp config YAML.")
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Chỉ định thư mục người cụ thể.")
@click.option("--apply", "-a", is_flag=True, default=False, help="Thực sự đổi tên chuẩn hóa.")
def cmd_clean(config: str, target_dir: Optional[str], apply: bool):
    """Chuẩn hóa tên tệp trong thư mục người thành dạng <tên>_<stt>.<ext>."""
    cfg = load_config(config)
    run_clean(cfg=cfg, target_dir=target_dir, apply=apply)


@cli.command("prepare")
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn tệp config YAML.")
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Thư mục đích lưu ảnh test.")
@click.option("--count", "-n", default=2, show_default=True, help="Số ảnh lấy từ mỗi người.")
@click.option("--copy/--move", default=True, help="Sao chép (an toàn) hoặc di chuyển.")
def cmd_prepare(config: str, target_dir: Optional[str], count: int, copy: bool):
    """Chuẩn bị dữ liệu kiểm thử từ backup."""
    cfg = load_config(config)
    run_prepare(cfg=cfg, target_dir=target_dir, count=count, copy=copy)


@cli.command("pipeline")
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Đường dẫn tệp config YAML.")
@click.option("--dir", "-d", "--target-dir", "target_dir", default=None, help="Thư mục xử lý.")
@click.option("--rename", "-r", is_flag=True, default=False, help="Chạy nhận diện & đổi tên.")
@click.option("--cluster", "-c-flag", "--cluster-faces", "do_cluster", is_flag=True, default=False, help="Chạy gom nhóm người lạ.")
@click.option("--organize", "-o", is_flag=True, default=False, help="Chạy di chuyển tệp về thư mục người.")
@click.option("--clean", "-n", is_flag=True, default=False, help="Chạy chuẩn hóa tên tệp.")
@click.option("--prepare", "-p", is_flag=True, default=False, help="Chạy chuẩn bị dữ liệu test.")
@click.option("--apply", "-a", is_flag=True, default=False, help="Thực sự áp dụng thay đổi.")
@click.option("--yes", "-y", is_flag=True, default=False, help="Tự động đồng ý tên mặc định.")
@click.option("--clear-cache", "-cc", is_flag=True, default=False, help="Xóa cache.")
def cmd_pipeline(
    config: str,
    target_dir: Optional[str],
    rename: bool,
    do_cluster: bool,
    organize: bool,
    clean: bool,
    prepare: bool,
    apply: bool,
    yes: bool,
    clear_cache: bool,
):
    """
    Chạy liên hoàn pipeline trong MỘT tiến trình duy nhất.
    Model và cache chỉ nạp 1 lần, tiết kiệm tối đa RAM và thời gian khởi động.
    """
    cfg = load_config(config)
    dataset_root = Path(cfg.dataset_root).expanduser().resolve()
    cache_path = dataset_root / cfg.cache_file
    cache = EmbeddingCache(cache_path)

    if clear_cache:
        cache.clear()
        click.echo("🗑️ Đã xóa cache.")

    if prepare:
        click.echo("\n🧪 [Bước 1] Chuẩn bị dữ liệu kiểm thử...")
        run_prepare(cfg=cfg, target_dir=target_dir, count=2, copy=True)

    # Khởi tạo FaceEngine dùng chung cho cả rename và cluster
    shared_engine = None
    if rename or do_cluster:
        shared_engine = FaceEngine(
            engine=cfg.engine,
            insightface_model=cfg.insightface_model,
            allowed_modules=cfg.allowed_modules,
            det_thresh=cfg.det_thresh,
            multi_face_strategy=cfg.multi_face_strategy,
            model=cfg.model,
            num_jitters=cfg.num_jitters,
            image_max_size=cfg.image_max_size,
        )

    if rename:
        click.echo("\n🔍 [Bước 2] Nhận diện và đổi tên...")
        run_rename(cfg=cfg, engine_obj=shared_engine, cache=cache, target_dir=target_dir, apply=apply)

    if do_cluster:
        click.echo("\n🤖 [Bước 3] Gom nhóm khuôn mặt người lạ...")
        run_cluster(cfg=cfg, engine_obj=shared_engine, cache=cache, target_dir=target_dir, apply=apply)

    if organize:
        click.echo("\n🚚 [Bước 4] Di chuyển tệp về thư mục người...")
        run_organize(cfg=cfg, target_dir=target_dir, apply=apply, yes=yes, include_low=True)

    if clean:
        click.echo("\n🔢 [Bước 5] Chuẩn hóa tên tệp trong thư mục...")
        run_clean(cfg=cfg, target_dir=target_dir, apply=apply)


if __name__ == "__main__":
    cli()
