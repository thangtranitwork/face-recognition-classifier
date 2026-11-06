"""Pipeline điều phối toàn bộ luồng xử lý: Rename, Cluster, Organize, Clean, Prepare."""

import os
import random
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN

from facecls.config import AppConfig
from facecls.io import (
    ALL_VALID_EXTENSIONS,
    VALID_IMAGE_EXTENSIONS,
    VALID_VIDEO_EXTENSIONS,
    resolve_target_dir,
    safe_rename,
    safe_move,
    build_new_name,
    is_already_processed,
    extract_candidate_prefix,
    load_image_rgb,
)
from facecls.cache import EmbeddingCache
from facecls.engine import FaceEngine
from facecls.matcher import KnownFaceIndex

logger = logging.getLogger("face_renamer")


def setup_logger(log_to_file: bool, log_file: str, dataset_root: Path) -> logging.Logger:
    """Thiết lập logger chuẩn cho toàn bộ ứng dụng."""
    log = logging.getLogger("face_renamer")
    log.setLevel(logging.DEBUG)

    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        log.addHandler(ch)

        if log_to_file:
            log_path = dataset_root / log_file
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            log.addHandler(fh)

    return log


def load_known_faces(
    cfg: AppConfig,
    engine_obj: FaceEngine,
    cache: EmbeddingCache,
    dataset_root: Path,
    target_unclassified: Optional[Path] = None,
) -> Tuple[Dict[str, List[np.ndarray]], KnownFaceIndex]:
    """
    Quét tập dữ liệu huấn luyện (Dataset), trích xuất và cache embedding theo từng tệp.
    Sử dụng random seed cố định để đảm bảo tính nhất quán tuyệt đối.
    """
    exclude_set = {d.lower() for d in cfg.exclude_dirs}
    if target_unclassified:
        exclude_set.add(target_unclassified.name.lower())

    person_dirs = sorted(
        [d for d in dataset_root.iterdir() if d.is_dir() and d.name.lower() not in exclude_set],
        key=lambda d: d.name.lower()
    )

    if not person_dirs:
        logger.error("❌ Không tìm thấy thư mục người nào trong dataset_root.")
        sys.exit(1)

    total_persons = len(person_dirs)
    logger.info(f"📂 Tìm thấy {total_persons} thư mục người: {[d.name for d in person_dirs]}")

    rng = random.Random(cfg.random_seed)
    known_faces: Dict[str, List[np.ndarray]] = {}
    cache_hits = 0
    new_encoded = 0

    for idx, p_dir in enumerate(person_dirs, start=1):
        person_name = p_dir.name
        media_files = sorted(
            [p for p in p_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS],
            key=lambda p: p.name.lower()
        )

        if not media_files:
            logger.warning(f"⚠️  [{idx}/{total_persons}] Thư mục '{person_name}' không có ảnh, bỏ qua.")
            continue

        # Lấy mẫu ngẫu nhiên có cố định seed
        shuffled = media_files.copy()
        rng.shuffle(shuffled)
        sample_files = shuffled[:cfg.max_images_per_person] if cfg.max_images_per_person else shuffled

        encodings = []
        for fpath in tqdm(sample_files, desc=f"  [{idx}/{total_persons}] Nạp {person_name}", leave=False, unit="img"):
            cached_enc = cache.get(fpath, cfg.engine, cfg.insightface_model)
            if cached_enc is not None:
                if cached_enc.size > 0:
                    encodings.append(cached_enc)
                cache_hits += 1
                continue

            # Chưa có cache -> trích xuất mới
            embs = engine_obj.extract_from_file(fpath, filter_quality=True, select_single=True)
            if embs:
                enc = embs[0]
                encodings.append(enc)
                cache.set(fpath, cfg.engine, cfg.insightface_model, enc)
            else:
                cache.set(fpath, cfg.engine, cfg.insightface_model, None)
            new_encoded += 1

        if encodings:
            known_faces[person_name] = encodings
            logger.info(f"  👤 [{idx}/{total_persons}] {person_name}: {len(encodings)} encodings (tổng {len(media_files)} ảnh)")
        else:
            logger.warning(f"⚠️  '{person_name}': Không phát hiện khuôn mặt hợp lệ nào.")

    cache.save()
    logger.info(f"⚡ Thống kê cache: {cache_hits} ảnh tái sử dụng | {new_encoded} ảnh mới mã hóa.")

    if not known_faces:
        logger.error("❌ Không có khuôn mặt đã biết nào hợp lệ trong dataset.")
        sys.exit(1)

    index = KnownFaceIndex(known_faces, engine=cfg.engine)
    return known_faces, index


def run_rename(
    cfg: AppConfig,
    engine_obj: Optional[FaceEngine] = None,
    cache: Optional[EmbeddingCache] = None,
    target_dir: Optional[str] = None,
    apply: bool = False,
) -> dict:
    """Tác vụ 1: Nhận diện khuôn mặt và đổi tên tệp."""
    dataset_root = Path(cfg.dataset_root).expanduser().resolve()
    unclassified_dir = resolve_target_dir(target_dir, cfg.unclassified_dir, dataset_root)

    logger = setup_logger(cfg.log_to_file, cfg.log_file, dataset_root)
    logger.info("=" * 60)
    logger.info("  Face Recognition File Renamer")
    logger.info("=" * 60)
    logger.info(f"📌 Nguồn         : {unclassified_dir}")
    logger.info(f"📌 Engine        : {cfg.engine} ({cfg.insightface_model})")
    logger.info(f"📌 Mode          : {'APPLY ✏️ (Đổi tên thật)' if apply else 'DRY-RUN 🔍 (Chỉ xem trước)'}")

    if cache is None:
        cache = EmbeddingCache(dataset_root / cfg.cache_file)
    if engine_obj is None:
        engine_obj = FaceEngine(
            engine=cfg.engine,
            insightface_model=cfg.insightface_model,
            allowed_modules=cfg.allowed_modules,
            det_thresh=cfg.det_thresh,
            multi_face_strategy=cfg.multi_face_strategy,
            model=cfg.model,
            num_jitters=cfg.num_jitters,
            image_max_size=cfg.image_max_size,
        )

    known_faces, index = load_known_faces(cfg, engine_obj, cache, dataset_root, unclassified_dir)

    media_files = sorted(
        [p for p in unclassified_dir.iterdir() if p.is_file() and p.suffix.lower() in ALL_VALID_EXTENSIONS],
        key=lambda p: p.name.lower()
    )

    if not media_files:
        logger.warning(f"⚠️ Không có tệp ảnh/video nào trong {unclassified_dir}")
        return {}

    logger.info(f"\n📁 Bắt đầu xử lý {len(media_files)} tệp media...")

    stats = {"recognized": 0, "low_confidence": 0, "unknown": 0, "skipped": 0, "error": 0}
    known_names = set(known_faces.keys())

    for fpath in tqdm(media_files, desc="Đang nhận diện", unit="file"):
        orig_name = fpath.name

        if is_already_processed(orig_name, known_names, cfg.unknown_prefix):
            logger.debug(f"  ⏭️ Bỏ qua (đã xử lý trước): {orig_name}")
            stats["skipped"] += 1
            continue

        ext = fpath.suffix.lower()
        if ext in VALID_VIDEO_EXTENSIONS:
            frames = engine_obj.extract_frames_from_video(fpath, sample_frames=cfg.sample_frames_per_video)
            person, reason, conf = index.match_video(
                frames=frames,
                engine_obj=engine_obj,
                top_k=cfg.top_k_neighbors,
                threshold=cfg.cosine_threshold if cfg.engine == "insightface" else cfg.tolerance,
                margin_threshold=cfg.margin_threshold,
                unknown_prefix=cfg.unknown_prefix,
            )
        else:
            cached_enc = cache.get(fpath, cfg.engine, cfg.insightface_model)
            if cached_enc is not None and cached_enc.size > 0:
                face_enc = cached_enc
            else:
                embs = engine_obj.extract_from_file(fpath, filter_quality=True, select_single=True)
                if embs:
                    face_enc = embs[0]
                    cache.set(fpath, cfg.engine, cfg.insightface_model, face_enc)
                else:
                    face_enc = None
                    cache.set(fpath, cfg.engine, cfg.insightface_model, None)

            if face_enc is None:
                person = cfg.unknown_prefix
                reason = "không phát hiện khuôn mặt hợp lệ"
                conf = 0.0
            else:
                person, reason, conf = index.match_single(
                    face_enc=face_enc,
                    top_k=cfg.top_k_neighbors,
                    threshold=cfg.cosine_threshold if cfg.engine == "insightface" else cfg.tolerance,
                    margin_threshold=cfg.margin_threshold,
                    unknown_prefix=cfg.unknown_prefix,
                )

        is_unknown = (person == cfg.unknown_prefix)
        is_low = (not is_unknown and conf < cfg.low_confidence_threshold)

        if is_unknown:
            fmt = cfg.name_format
            stats["unknown"] += 1
            log_icon = "❓"
        elif is_low:
            fmt = cfg.low_confidence_format
            stats["low_confidence"] += 1
            log_icon = "⚠️ "
        else:
            fmt = cfg.name_format
            stats["recognized"] += 1
            log_icon = "✅"

        new_name = build_new_name(person, orig_name, fmt)
        logger.info(f"  {log_icon} {orig_name} → {new_name}  [{reason}]")

        if apply:
            try:
                safe_rename(fpath, new_name)
            except Exception as e:
                logger.error(f"  ❌ Lỗi đổi tên {orig_name}: {e}")
                stats["error"] += 1

    cache.save()

    logger.info("\n" + "=" * 60)
    logger.info("  KẾT QUẢ ĐỔI TÊN")
    logger.info("=" * 60)
    logger.info(f"  ✅ Nhận diện được      : {stats['recognized']}")
    logger.info(f"  ⚠️  Độ tin cậy thấp    : {stats['low_confidence']} (< {cfg.low_confidence_threshold}%)")
    logger.info(f"  ❓ Không nhận diện (Unknown): {stats['unknown']}")
    logger.info(f"  ⏭️  Đã xử lý trước     : {stats['skipped']}")
    logger.info(f"  ❌ Lỗi đổi tên         : {stats['error']}")
    if not apply:
        logger.info("\n  💡 Chạy với '--apply' (hoặc '-a') để thực sự áp dụng đổi tên.")
    logger.info("=" * 60)

    return stats


def run_cluster(
    cfg: AppConfig,
    engine_obj: Optional[FaceEngine] = None,
    cache: Optional[EmbeddingCache] = None,
    target_dir: Optional[str] = None,
    apply: bool = False,
    override_eps: Optional[float] = None,
    override_min_samples: Optional[int] = None,
    workers: int = 4,
) -> None:
    """Tác vụ 2: Gom nhóm tự động khuôn mặt người lạ (DBSCAN)."""
    dataset_root = Path(cfg.dataset_root).expanduser().resolve()
    unclassified_dir = resolve_target_dir(target_dir, cfg.unclassified_dir, dataset_root)

    click.echo("=" * 60)
    click.echo("  Auto-Clustering Unknown Faces (Gom nhóm người lạ)")
    click.echo("=" * 60)

    if cache is None:
        cache = EmbeddingCache(dataset_root / cfg.cache_file)
    if engine_obj is None:
        engine_obj = FaceEngine(
            engine=cfg.engine,
            insightface_model=cfg.insightface_model,
            allowed_modules=cfg.allowed_modules,
            det_thresh=cfg.det_thresh,
            multi_face_strategy=cfg.multi_face_strategy,
            model=cfg.model,
            num_jitters=cfg.num_jitters,
            image_max_size=cfg.image_max_size,
        )

    # Chọn khoảng cách và metric phù hợp với engine
    metric = "cosine" if cfg.engine == "insightface" else "euclidean"
    default_eps = cfg.cluster_eps_cosine if cfg.engine == "insightface" else cfg.cluster_eps_dlib
    dbscan_eps = override_eps if override_eps is not None else default_eps
    min_samples = override_min_samples if override_min_samples is not None else cfg.cluster_min_samples

    exclude_dirs = {d.lower() for d in cfg.exclude_dirs}
    exclude_dirs.add(unclassified_dir.name.lower())
    known_person_names = {
        d.name.lower() for d in dataset_root.iterdir()
        if d.is_dir() and d.name.lower() not in exclude_dirs
    }

    # Chỉ quét tệp CHƯA phân loại (unknown hoặc chưa gán nhãn người đã biết)
    candidate_files = []
    for p in unclassified_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in ALL_VALID_EXTENSIONS:
            continue
        n_lower = p.name.lower()
        if n_lower.startswith("low_"):
            continue
        is_known = False
        for p_name in known_person_names:
            if n_lower.startswith(f"{p_name}_") or n_lower.startswith(f"{p_name}-"):
                is_known = True
                break
        if not is_known:
            candidate_files.append(p)

    candidate_files.sort(key=lambda p: p.name.lower())
    if not candidate_files:
        click.echo(f"⚠️  Không có tệp chưa phân loại nào trong '{unclassified_dir.name}/'.")
        return

    click.echo(f"📌 Mode          : {'APPLY ✏️ (Gom nhóm & đổi tên file)' if apply else 'DRY-RUN 🔍 (Chỉ xem trước)'}")
    click.echo(f"📌 Nguồn         : {unclassified_dir}")
    click.echo(f"📌 Engine        : {cfg.engine} ({cfg.insightface_model})")
    click.echo(f"📌 Metric        : {metric} | DBSCAN eps: {dbscan_eps} | Min samples: {min_samples}")
    click.echo(f"📌 Số tệp quét   : {len(candidate_files)}")

    # Trích xuất vector: Tái sử dụng triệt để từ cache
    encodings_list = []
    valid_file_paths = []
    files_to_extract = []

    for fpath in candidate_files:
        cached_enc = cache.get(fpath, cfg.engine, cfg.insightface_model)
        if cached_enc is not None:
            if cached_enc.size > 0:
                encodings_list.append(cached_enc)
                valid_file_paths.append(fpath)
        else:
            files_to_extract.append(fpath)

    if files_to_extract:
        click.echo(f"🔍 Đang trích xuất đặc trưng cho {len(files_to_extract)} tệp mới...")
        # Đọc ảnh song song ở tầng I/O để tối ưu tốc độ
        def _read_worker(p: Path):
            return p, load_image_rgb(p, max_size=cfg.image_max_size)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_path = {executor.submit(_read_worker, p): p for p in files_to_extract}
            for fut in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Trích xuất", unit="file"):
                p, img_rgb = fut.result()
                if img_rgb is not None:
                    embs = engine_obj.extract_face_embeddings(img_rgb, filter_quality=True, select_single=True)
                    if embs:
                        enc = embs[0]
                        cache.set(p, cfg.engine, cfg.insightface_model, enc)
                        encodings_list.append(enc)
                        valid_file_paths.append(p)
                        continue
                cache.set(p, cfg.engine, cfg.insightface_model, None)

        cache.save()

    if not encodings_list:
        click.echo("⚠️ Không trích xuất được khuôn mặt hợp lệ nào từ các tệp cần gom nhóm.")
        return

    click.echo(f"✅ Đã có {len(encodings_list)} vector khuôn mặt. Đang tiến hành gom nhóm (DBSCAN)...")

    X = np.vstack(encodings_list)
    clt = DBSCAN(eps=dbscan_eps, min_samples=min_samples, metric=metric)
    labels = clt.fit_predict(X)

    unique_labels = sorted(list({l for l in labels if l != -1}))
    if not unique_labels:
        click.echo("\n⚠️  Không tìm thấy nhóm người mới nào xuất hiện trùng lặp (tất cả là khuôn mặt đơn lẻ).")
        return

    click.echo(f"\n🎉 Phát hiện {len(unique_labels)} nhóm người mới xuất hiện trùng lặp:\n")

    # Xác định chỉ số tiếp theo cho tên nhóm (vd: nguoi_moi_1)
    existing_dirs = {d.name.lower() for d in dataset_root.iterdir() if d.is_dir()}
    next_idx = 1
    while f"{cfg.cluster_prefix.lower()}_{next_idx}" in existing_dirs:
        next_idx += 1

    for idx_c, cluster_id in enumerate(unique_labels, start=1):
        indices = [i for i, l in enumerate(labels) if l == cluster_id]
        c_files = [valid_file_paths[i] for i in indices]
        c_encs = X[indices]

        # Tính độ tương đồng trung bình nội nhóm
        if len(c_encs) > 1:
            sim_matrix = np.dot(c_encs, c_encs.T)
            triu = np.triu_indices(len(c_encs), k=1)
            avg_sim = float(np.mean(sim_matrix[triu]))
            avg_sim = float(np.clip(avg_sim, -1.0, 1.0))
        else:
            avg_sim = 1.0

        group_name = f"{cfg.cluster_prefix}_{next_idx}"
        click.echo(
            f"👤 [{idx_c}/{len(unique_labels)}] Nhóm '{group_name}' "
            f"({len(c_files)} tệp | Độ tương đồng: {avg_sim * 100:.1f}%):"
        )
        for f in c_files:
            click.echo(f"   • {f.name}")

        if apply:
            for f in c_files:
                c_name = f.name
                if c_name.lower().startswith(f"{cfg.unknown_prefix.lower()}_"):
                    c_name = c_name[len(cfg.unknown_prefix) + 1:]
                new_name = f"{group_name}_{c_name}"
                safe_rename(f, new_name)
            click.echo(f"   ✅ Đã đổi tên {len(c_files)} tệp thành {group_name}_*")

        next_idx += 1

    click.echo("\n" + "=" * 60)
    click.echo(f"  Nhóm người mới tìm thấy : {len(unique_labels)}")
    click.echo(f"  Số tệp đã gom nhóm      : {sum(1 for l in labels if l != -1)}")
    click.echo(f"  Số tệp đơn lẻ (noise)   : {sum(1 for l in labels if l == -1)}")
    if not apply:
        click.echo("  💡 Chạy với '--apply' (hoặc '-a') để thực sự đổi tên các tệp đã gom nhóm.")
    click.echo("=" * 60)


def run_organize(
    cfg: AppConfig,
    target_dir: Optional[str] = None,
    apply: bool = False,
    yes: bool = False,
    include_low: bool = True,
) -> None:
    """Tác vụ 3: Di chuyển các tệp đã đổi tên về đúng thư mục người tương ứng."""
    dataset_root = Path(cfg.dataset_root).expanduser().resolve()
    unclassified_dir = resolve_target_dir(target_dir, cfg.unclassified_dir, dataset_root)

    click.echo("=" * 60)
    click.echo("  Organize Renamed Faces Tool")
    click.echo("=" * 60)
    click.echo(f"📌 Mode          : {'APPLY ✏️ (Di chuyển thật)' if apply else 'DRY-RUN 🔍 (Chỉ xem trước)'}")
    click.echo(f"📌 Nguồn         : {unclassified_dir}")
    click.echo(f"📌 Loại trừ      : {cfg.unknown_prefix}_* (giữ nguyên không chuyển)")
    click.echo(f"📌 Bao gồm low   : {'Có' if include_low else 'Không'}")
    click.echo("")

    exclude_dirs = {d.lower() for d in cfg.exclude_dirs}
    exclude_dirs.add(unclassified_dir.name.lower())

    person_dirs: Dict[str, Path] = {
        d.name.lower(): d for d in dataset_root.iterdir()
        if d.is_dir() and d.name.lower() not in exclude_dirs
    }

    images = [
        p for p in unclassified_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ALL_VALID_EXTENSIONS
    ]

    if not images:
        click.echo(f"⚠️  Không có tệp nào trong '{unclassified_dir.name}/'")
        return

    # 1. Thu thập các tệp chưa khớp thư mục hiện có
    unmatched_files = []
    for img in sorted(images, key=lambda p: p.name.lower()):
        n_lower = img.name.lower()
        if n_lower.startswith(f"{cfg.unknown_prefix.lower()}_") or n_lower.startswith(f"{cfg.unknown_prefix.lower()}-"):
            continue

        clean_name = img.name[4:] if (n_lower.startswith("low_") or n_lower.startswith("low-")) else img.name
        c_lower = clean_name.lower()

        matched = any(
            c_lower.startswith(f"{p}_") or c_lower.startswith(f"{p}-")
            for p in person_dirs.keys()
        )
        if not matched:
            unmatched_files.append(img)

    # 2. Gom nhóm unmatched theo prefix người mới (vd: nguoi_moi_1)
    new_groups: Dict[str, List[Path]] = {}
    for img in unmatched_files:
        prefix = extract_candidate_prefix(img.name, cfg.unknown_prefix)
        if prefix:
            new_groups.setdefault(prefix, []).append(img)

    unregistered = [(p, files) for p, files in new_groups.items() if p.lower() not in person_dirs]

    # Xử lý hỏi tên người thật cho nhóm người mới
    for idx_g, (prefix, g_files) in enumerate(unregistered, start=1):
        click.echo("─" * 60)
        click.echo(f"👤 [{idx_g}/{len(unregistered)}] Nhóm người mới '{prefix}' ({len(g_files)} tệp):")
        for f in g_files:
            click.echo(f"   • {f.name}")

        default_name = prefix.lower()
        if apply:
            if yes or not sys.stdin.isatty():
                chosen_name = default_name
            else:
                chosen_name = click.prompt(
                    f"👉 Nhập tên người thật cho nhóm '{prefix}' (Enter để giữ nguyên)",
                    default=default_name,
                    show_default=True
                ).strip()

            chosen_name = (chosen_name or default_name).lower()
            new_dir = dataset_root / chosen_name
            new_dir.mkdir(exist_ok=True, parents=True)
            person_dirs[chosen_name] = new_dir

            # Đổi lại prefix tên tệp nếu người dùng đặt tên khác prefix tạm thời
            if chosen_name != prefix.lower():
                for f in g_files:
                    orig_name = f.name
                    if orig_name.lower().startswith("low_"):
                        rest = orig_name[4 + len(prefix) + 1:]
                        new_fname = f"low_{chosen_name}_{rest}"
                    elif orig_name.lower().startswith(f"{prefix.lower()}_"):
                        rest = orig_name[len(prefix) + 1:]
                        new_fname = f"{chosen_name}_{rest}"
                    else:
                        new_fname = f"{chosen_name}_{orig_name}"
                    safe_rename(f, new_fname)

            click.echo(f"   ✅ Đã tạo thư mục dataset/{chosen_name}/")
        else:
            person_dirs[prefix.lower()] = dataset_root / default_name
            click.echo(f"   💡 Chạy với '-a' để gán tên người thật và tạo thư mục dataset/{default_name}/")

    # 3. Quét lại và di chuyển tệp vào đúng thư mục
    images = [
        p for p in unclassified_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ALL_VALID_EXTENSIONS
    ]

    click.echo("\n" + "=" * 60)
    click.echo("  TIẾN HÀNH DI CHUYỂN TỆP")
    click.echo("=" * 60)

    moved_count = 0
    skipped_unknown = 0
    unmatched_count = 0

    for img in sorted(images, key=lambda p: p.name.lower()):
        n_lower = img.name.lower()
        if n_lower.startswith(f"{cfg.unknown_prefix.lower()}_") or n_lower.startswith(f"{cfg.unknown_prefix.lower()}-"):
            click.echo(f"  ⏭️  Giữ nguyên (Unknown): {img.name}")
            skipped_unknown += 1
            continue

        is_low = n_lower.startswith("low_") or n_lower.startswith("low-")
        clean_name = img.name[4:] if is_low else img.name

        if is_low and not include_low:
            click.echo(f"  ⏭️  Bỏ qua (Low confidence): {img.name}")
            continue

        c_lower = clean_name.lower()
        matched_target_dir = None
        for p_name, p_dir in sorted(person_dirs.items(), key=lambda x: len(x[0]), reverse=True):
            if c_lower.startswith(f"{p_name}_") or c_lower.startswith(f"{p_name}-"):
                matched_target_dir = p_dir
                break

        if not matched_target_dir:
            click.echo(f"  ⚠️  Không khớp thư mục: {img.name}")
            unmatched_count += 1
            continue

        tag = "⚠️  LOW" if is_low else "✅ HIGH"
        click.echo(f"  🚚 [{tag}] {img.name} → {matched_target_dir.name}/")

        if apply:
            try:
                safe_move(img, matched_target_dir)
                moved_count += 1
            except Exception as e:
                click.echo(f"  ❌ Lỗi di chuyển {img.name}: {e}", err=True)
        else:
            moved_count += 1

    click.echo("\n" + "=" * 60)
    click.echo(f"  🚚 Tệp đã di chuyển   : {moved_count}")
    click.echo(f"  ⏭️  Tệp giữ lại       : {skipped_unknown}")
    click.echo(f"  ⚠️  Tệp không khớp     : {unmatched_count}")
    if not apply:
        click.echo("\n  💡 Chạy với '--apply' (hoặc '-a') để thực sự di chuyển tệp.")
    click.echo("=" * 60)


def run_clean(
    cfg: AppConfig,
    target_dir: Optional[str] = None,
    apply: bool = False,
) -> None:
    """Tác vụ 4: Chuẩn hóa tên tệp trong thư mục người thành dạng chuẩn <tên>_<stt>.<ext>."""
    dataset_root = Path(cfg.dataset_root).expanduser().resolve()
    unclassified_name = cfg.unclassified_dir
    exclude_dirs = {d.lower() for d in cfg.exclude_dirs}
    exclude_dirs.add(unclassified_name.lower())

    if target_dir:
        p = resolve_target_dir(target_dir, unclassified_name, dataset_root)
        unclass_path = dataset_root / unclassified_name
        if p.name.lower() in exclude_dirs or p == unclass_path or unclass_path in p.parents:
            click.echo(f"❌ Thư mục '{p.name}' bị loại trừ. Không chạy chuẩn hóa lên thư mục này.", err=True)
            sys.exit(1)
        person_dirs = [p]
    else:
        person_dirs = [
            d for d in dataset_root.iterdir()
            if d.is_dir() and d.name.lower() not in exclude_dirs
        ]

    person_dirs.sort(key=lambda d: d.name.lower())
    if not person_dirs:
        click.echo("⚠️ Không tìm thấy thư mục người nào để chuẩn hóa.")
        return

    click.echo("=" * 60)
    click.echo("  Normalize Dataset File Names (Tên + STT)")
    click.echo("=" * 60)
    click.echo(f"📌 Mode          : {'APPLY ✏️ (Đổi tên thật)' if apply else 'DRY-RUN 🔍 (Chỉ xem trước)'}")
    click.echo(f"📌 Thư mục xử lý : {len(person_dirs)}")
    click.echo("")

    total_renamed = 0

    for idx, p_dir in enumerate(person_dirs, start=1):
        p_name = p_dir.name
        files = sorted(
            [f for f in p_dir.iterdir() if f.is_file() and f.suffix.lower() in ALL_VALID_EXTENSIONS],
            key=lambda f: f.name.lower()
        )
        if not files:
            continue

        click.echo(f"📁 [{idx}/{len(person_dirs)}] [{p_name}] {len(files)} tệp:")

        # Bước A: Đổi sang tên tạm trung gian để tránh xung đột
        pairs = []
        for i, f in enumerate(files, start=1):
            target_name = f"{p_name}_{i}{f.suffix.lower()}"
            pairs.append((f, target_name))
            click.echo(f"   • {f.name} → {target_name}")

        if apply:
            temp_pairs = []
            for i, (src, target_name) in enumerate(pairs, start=1):
                temp_name = f"__temp_norm_{i}__{src.name}"
                temp_path = src.parent / temp_name
                try:
                    src.rename(temp_path)
                    temp_pairs.append((temp_path, target_name))
                except Exception as e:
                    click.echo(f"   ❌ Lỗi đổi tên tạm {src.name}: {e}", err=True)

            for temp_path, target_name in temp_pairs:
                final_path = temp_path.parent / target_name
                try:
                    temp_path.rename(final_path)
                    total_renamed += 1
                except Exception as e:
                    click.echo(f"   ❌ Lỗi đổi tên chính thức: {e}", err=True)
        else:
            total_renamed += len(pairs)

    click.echo("\n" + "=" * 60)
    click.echo(f"  ✏️ Tổng số tệp {'đã đổi tên' if apply else 'sẽ đổi tên'}: {total_renamed}")
    if not apply:
        click.echo("  💡 Chạy với '--apply' (hoặc '-a') để thực sự chuẩn hóa tên tệp.")
    click.echo("=" * 60)


def run_prepare(
    cfg: AppConfig,
    target_dir: Optional[str] = None,
    count: int = 3,
    copy: bool = True,
    include_other: bool = True,
) -> None:
    """Tác vụ 5: Khôi phục từ backup và chuẩn bị dữ liệu kiểm thử an toàn."""
    dataset_root = Path(cfg.dataset_root).expanduser().resolve()
    unclassified_name = cfg.unclassified_dir

    backup_dir = dataset_root.parent / "dataset-backup"
    if not backup_dir.exists():
        backup_dir = dataset_root.parent / "dataset_backup"

    if backup_dir.exists():
        click.echo(f"🗑️ Khôi phục dataset sạch từ backup '{backup_dir.name}'...")
        import shutil
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        shutil.copytree(backup_dir, dataset_root)
        click.echo("✅ Đã khôi phục dataset hoàn chỉnh từ backup!\n")
    else:
        click.echo(f"⚠️ Không tìm thấy thư mục backup ('dataset_backup').")
        # Bảo vệ dữ liệu huấn luyện: Không tự ý di chuyển (move) khi không có bản backup
        if not copy:
            click.echo("🛡️ Để bảo vệ dữ liệu huấn luyện khi chưa có backup, chế độ COPY được kích hoạt tự động.")
            copy = True

    dest_dir = resolve_target_dir(target_dir, unclassified_name, dataset_root)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Dọn dẹp sạch sẽ thư mục đích trước khi tạo dữ liệu test mới
    for old_file in dest_dir.iterdir():
        if old_file.is_file() and old_file.suffix.lower() in ALL_VALID_EXTENSIONS:
            try:
                old_file.unlink()
            except Exception:
                pass

    exclude_dirs = {d.lower() for d in cfg.exclude_dirs}
    exclude_dirs.add(dest_dir.name.lower())

    person_dirs = sorted([
        d for d in dataset_root.iterdir()
        if d.is_dir() and d.name.lower() not in exclude_dirs
    ], key=lambda d: d.name.lower())

    if not person_dirs:
        click.echo("❌ Không tìm thấy thư mục người nào để lấy mẫu kiểm thử.", err=True)
        sys.exit(1)

    action = "Copying" if copy else "Moving"
    click.echo(f"📦 {action} ngẫu nhiên {count} tệp từ mỗi thư mục người sang '{dest_dir.name}/'...")

    rng = random.Random(cfg.random_seed)
    counter = 1
    total_processed = 0

    for p_dir in person_dirs:
        files = [f for f in p_dir.iterdir() if f.is_file() and f.suffix.lower() in ALL_VALID_EXTENSIONS]
        if not files:
            continue
        selected = rng.sample(files, min(count, len(files)))
        for src in selected:
            dest_name = f"IMG_{counter:04d}{src.suffix.lower()}"
            dest_path = dest_dir / dest_name
            import shutil
            if copy:
                shutil.copy2(src, dest_path)
            else:
                shutil.move(str(src), str(dest_path))
            click.echo(f"  {'📋' if copy else '🚚'} {p_dir.name}/{src.name} → {dest_dir.name}/{dest_name}")
            counter += 1
            total_processed += 1

    # Bổ sung ảnh người lạ từ thư mục 'other' (nếu có) để kiểm thử thuật toán gom nhóm Unknown faces
    other_dir = dataset_root / "other"
    if include_other and other_dir.exists() and other_dir.is_dir():
        other_files = [f for f in other_dir.iterdir() if f.is_file() and f.suffix.lower() in ALL_VALID_EXTENSIONS]
        if other_files:
            click.echo(f"\n🎭 Bổ sung {len(other_files)} ảnh người lạ (Unknown) từ 'other/' để kiểm thử gom nhóm...")
            for src in other_files:
                dest_name = f"IMG_{counter:04d}{src.suffix.lower()}"
                dest_path = dest_dir / dest_name
                import shutil
                shutil.copy2(src, dest_path)
                click.echo(f"  📋 other/{src.name} → {dest_dir.name}/{dest_name} (Unknown)")
                counter += 1
                total_processed += 1

    click.echo(f"\n✅ Đã chuẩn bị {total_processed} tệp kiểm thử trong '{dest_dir.name}/'.")
