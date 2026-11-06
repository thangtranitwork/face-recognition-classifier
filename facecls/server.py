"""FastAPI Web Server cho Face Recognition Studio."""

import os
import sys
import json
import asyncio
import re
import time
import base64
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
import numpy as np
import cv2
from PIL import Image

import urllib.request
import urllib.parse
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import yaml

from facecls.config import load_config, AppConfig
from facecls.io import (
    ALL_VALID_EXTENSIONS,
    VALID_IMAGE_EXTENSIONS,
    resolve_target_dir,
    safe_rename,
    safe_move,
    build_new_name,
    is_already_processed,
    load_image_rgb,
    generate_unique_path,
)
from facecls.cache import EmbeddingCache
from facecls.engine import FaceEngine
from facecls.matcher import KnownFaceIndex
from facecls.pipeline import load_known_faces, run_organize, run_clean, run_prepare
from facecls.downloader import (
    search_image_urls_page,
    search_instagram_images,
    iter_instagram_images,
    iter_instagram_highlights,
    resolve_instagram_profile_info,
    search_facebook_images,
    iter_facebook_images,
    resolve_facebook_profile_info,
    download_and_verify_image,
    preload_anchor_from_dir,
)
from facecls.video import (
    is_video_url,
    download_video_stream,
    sample_video_frames,
    analyze_video_motion_and_best_frame,
)

app = FastAPI(title="Face Recognition Studio", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "facecls" / "static"
TEMPLATES_DIR = PROJECT_ROOT / "facecls" / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Shared Engine and Cache instance
_CACHED_ENGINE: Optional[FaceEngine] = None
_CACHED_INDEX: Optional[KnownFaceIndex] = None
_CACHED_CACHE: Optional[EmbeddingCache] = None
ACTIVE_CANCELLATIONS: Set[str] = set()


def get_shared_engine() -> FaceEngine:
    global _CACHED_ENGINE
    if _CACHED_ENGINE is None:
        cfg = load_config()
        _CACHED_ENGINE = FaceEngine(
            engine=cfg.engine,
            insightface_model=cfg.insightface_model,
            allowed_modules=cfg.allowed_modules,
            det_thresh=cfg.det_thresh,
            multi_face_strategy=cfg.multi_face_strategy,
            image_max_size=cfg.image_max_size,
        )
    return _CACHED_ENGINE


def get_shared_cache() -> EmbeddingCache:
    global _CACHED_CACHE
    if _CACHED_CACHE is None:
        cfg = load_config()
        cache_path = PROJECT_ROOT / cfg.dataset_root / cfg.cache_file
        _CACHED_CACHE = EmbeddingCache(cache_path)
    return _CACHED_CACHE


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Face Recognition Studio UI</h1><p>Đang tải tệp template...</p>")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


# ────────────────────────────────────────────────────────────
# Media Serving
# ────────────────────────────────────────────────────────────

@app.get("/api/media/{file_path:path}")
@app.head("/api/media/{file_path:path}")
async def serve_media(file_path: str):
    """Phục vụ file ảnh/video an toàn."""
    target_path = (PROJECT_ROOT / file_path).resolve()
    # Kiểm tra an toàn đường dẫn phải nằm trong thư mục dự án
    if not str(target_path).startswith(str(PROJECT_ROOT)):
        raise HTTPException(status_code=403, detail="Truy cập bị từ chối")
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp")

    suffix = target_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    return FileResponse(target_path, media_type=media_type)


@app.get("/api/proxy_image")
@app.head("/api/proxy_image")
async def proxy_image(url: str = Query(...)):
    """Proxy ảnh an toàn để hiển thị trên UI mà không bị chặn bởi CORS/Referer của Instagram/Facebook CDN."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as res:
            content_type = res.headers.get("content-type", "image/jpeg")
            data = res.read()
            return Response(content=data, media_type=content_type)
    except Exception as e:
        print(f"[PROXY ERROR] for URL '{url[:80]}...': {e}")
        raise HTTPException(status_code=404, detail=f"Không thể tải ảnh proxy: {e}")


# ────────────────────────────────────────────────────────────
# Stats & Dataset Explorer
# ────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    cfg = load_config()
    dataset_root = (PROJECT_ROOT / cfg.dataset_root).resolve()
    exclude_dirs = {d.lower() for d in cfg.exclude_dirs}

    person_dirs = [
        d for d in dataset_root.iterdir()
        if d.is_dir() and d.name.lower() not in exclude_dirs
    ]

    total_training_images = sum(
        1 for d in person_dirs
        for f in d.iterdir()
        if f.is_file() and f.suffix.lower() in ALL_VALID_EXTENSIONS
    )

    unclassified_dir = dataset_root / cfg.unclassified_dir
    unclassified_count = sum(
        1 for f in unclassified_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ALL_VALID_EXTENSIONS
    ) if unclassified_dir.exists() else 0

    download_dir = PROJECT_ROOT / "download"
    download_count = sum(
        1 for f in download_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in ALL_VALID_EXTENSIONS
    ) if download_dir.exists() else 0

    cache = get_shared_cache()

    return {
        "total_persons": len(person_dirs),
        "total_training_images": total_training_images,
        "unclassified_count": unclassified_count,
        "download_count": download_count,
        "cache_entries": len(cache.embedding_store),
        "engine": cfg.engine,
        "model": cfg.insightface_model,
        "cosine_threshold": cfg.cosine_threshold,
    }


@app.get("/api/persons")
async def list_persons():
    cfg = load_config()
    dataset_root = (PROJECT_ROOT / cfg.dataset_root).resolve()
    exclude_dirs = {d.lower() for d in cfg.exclude_dirs}

    person_dirs = sorted([
        d for d in dataset_root.iterdir()
        if d.is_dir() and d.name.lower() not in exclude_dirs
    ], key=lambda d: d.name.lower())

    results = []
    for d in person_dirs:
        files = [
            f for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in ALL_VALID_EXTENSIONS
        ]
        avatar_url = None
        if files:
            files_sorted = sorted(files, key=lambda f: f.name.lower())
            rel = files_sorted[0].relative_to(PROJECT_ROOT)
            avatar_url = f"/api/media/{rel}"

        results.append({
            "name": d.name,
            "count": len(files),
            "avatar_url": avatar_url,
            "path": str(d.relative_to(PROJECT_ROOT)),
        })

    return results


@app.get("/api/persons/{person_name}")
async def get_person_details(person_name: str):
    cfg = load_config()
    dataset_root = (PROJECT_ROOT / cfg.dataset_root).resolve()
    target_dir = dataset_root / person_name
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=404, detail="Không tìm thấy thư mục người này")

    files = sorted([
        f for f in target_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ALL_VALID_EXTENSIONS
    ], key=lambda f: f.name.lower())

    items = []
    for f in files:
        rel = f.relative_to(PROJECT_ROOT)
        items.append({
            "name": f.name,
            "path": str(rel),
            "url": f"/api/media/{rel}",
            "size": f.stat().st_size,
        })

    return {
        "person": person_name,
        "total_files": len(items),
        "files": items,
    }


@app.get("/api/unclassified")
async def list_unclassified():
    cfg = load_config()
    dataset_root = (PROJECT_ROOT / cfg.dataset_root).resolve()
    target_dir = dataset_root / cfg.unclassified_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([
        f for f in target_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ALL_VALID_EXTENSIONS
    ], key=lambda f: f.name.lower())

    items = []
    for f in files:
        rel = f.relative_to(PROJECT_ROOT)
        items.append({
            "name": f.name,
            "path": str(rel),
            "url": f"/api/media/{rel}",
            "size": f.stat().st_size,
        })

    return {
        "total": len(items),
        "files": items,
    }


@app.post("/api/upload")
async def upload_files(
    target_dir: str = Form("chua_phan_loai"),
    files: List[UploadFile] = File(...),
):
    cfg = load_config()
    dataset_root = (PROJECT_ROOT / cfg.dataset_root).resolve()

    if target_dir == "chua_phan_loai":
        dest_dir = dataset_root / cfg.unclassified_dir
    else:
        dest_dir = dataset_root / target_dir.lower().strip()

    dest_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for upload in files:
        fname = upload.filename or "upload.jpg"
        save_path = generate_unique_path(dest_dir, fname)
        content = await upload.read()
        with open(save_path, "wb") as f:
            f.write(content)
        saved_files.append(save_path.name)

    return {"uploaded": len(saved_files), "files": saved_files}


@app.delete("/api/media")
async def delete_media(payload: Dict[str, str]):
    path_str = payload.get("path")
    if not path_str:
        raise HTTPException(status_code=400, detail="Thiếu đường dẫn tệp")
    target_path = (PROJECT_ROOT / path_str).resolve()
    if not str(target_path).startswith(str(PROJECT_ROOT)):
        raise HTTPException(status_code=403, detail="Không có quyền truy cập")
    if target_path.exists() and target_path.is_file():
        target_path.unlink()
        return {"success": True, "deleted": target_path.name}
    raise HTTPException(status_code=404, detail="Tệp không tồn tại")


# ────────────────────────────────────────────────────────────
# Scan, Recognition & Renaming
# ────────────────────────────────────────────────────────────

@app.post("/api/scan")
async def scan_unclassified(payload: Dict[str, Any] = {}):
    apply = payload.get("apply", False)
    cfg = load_config()
    dataset_root = (PROJECT_ROOT / cfg.dataset_root).resolve()
    unclassified_dir = dataset_root / cfg.unclassified_dir

    engine_obj = get_shared_engine()
    cache = get_shared_cache()

    known_faces, index = load_known_faces(cfg, engine_obj, cache, dataset_root, unclassified_dir)

    media_files = sorted([
        f for f in unclassified_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ALL_VALID_EXTENSIONS
    ], key=lambda f: f.name.lower())

    stats = {
        "recognized": 0,
        "low_confidence": 0,
        "unknown": 0,
        "skipped": 0,
        "renamed": 0,
        "error": 0,
        "total": len(media_files),
    }
    results = []
    known_names = set(known_faces.keys())
    now_ts = int(time.time() * 1000)

    force = payload.get("force", False)

    for fpath in media_files:
        orig_name = fpath.name
        rel = fpath.relative_to(PROJECT_ROOT)
        media_url = f"/api/media/{rel}?t={now_ts}"

        is_already = is_already_processed(orig_name, known_names, cfg.unknown_prefix)

        # Nếu file đã đổi tên trước đó và không yêu cầu quét cưỡng bức, bỏ qua bước inference ONNX nặng
        if is_already and not force and not apply:
            stats["skipped"] += 1
            predicted_p = orig_name.split("_")[0]
            results.append({
                "original": orig_name,
                "current_name": orig_name,
                "new_name": orig_name,
                "url": media_url,
                "person": predicted_p,
                "confidence": 100.0,
                "reason": f"Đã được gắn nhãn '{predicted_p}'",
                "status": "already_named",
                "bbox": None,
            })
            continue

        # Trích xuất bounding box và vector đặc trưng (để trực quan hóa trên giao diện)
        img_rgb = load_image_rgb(fpath, max_size=cfg.image_max_size)
        bbox = None
        face_enc = None

        if img_rgb is not None:
            h, w, _ = img_rgb.shape
            faces = engine_obj.app.get(img_rgb[:, :, ::-1]) if engine_obj.engine == "insightface" else []
            if faces:
                best_face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
                b = best_face.bbox
                bbox = {
                    "x": float(b[0] / w),
                    "y": float(b[1] / h),
                    "w": float((b[2] - b[0]) / w),
                    "h": float((b[3] - b[1]) / h),
                    "score": round(float(best_face.det_score) * 100, 1),
                }
                emb = best_face.embedding
                norm = np.linalg.norm(emb)
                if norm > 0:
                    face_enc = (emb / norm).astype(np.float32)

        if is_already and not apply:
            stats["skipped"] += 1
            predicted_p = orig_name.split("_")[0]
            conf = 100.0
            if face_enc is not None:
                _, _, c_val = index.match_single(face_enc, top_k=cfg.top_k_neighbors)
                conf = c_val

            results.append({
                "original": orig_name,
                "current_name": orig_name,
                "new_name": orig_name,
                "url": media_url,
                "person": predicted_p,
                "confidence": conf,
                "reason": f"Đã được gắn nhãn '{predicted_p}'",
                "status": "already_named",
                "bbox": bbox,
            })
            continue

        if face_enc is None:
            person = cfg.unknown_prefix
            reason = "Không phát hiện khuôn mặt"
            conf = 0.0
        else:
            person, reason, conf = index.match_single(
                face_enc=face_enc,
                top_k=cfg.top_k_neighbors,
                threshold=cfg.cosine_threshold,
                margin_threshold=cfg.margin_threshold,
                unknown_prefix=cfg.unknown_prefix,
            )

        is_unknown = (person == cfg.unknown_prefix)
        is_low = (not is_unknown and conf < cfg.low_confidence_threshold)

        if is_unknown:
            stats["unknown"] += 1
            status = "unknown"
            fmt = cfg.name_format
        elif is_low:
            stats["low_confidence"] += 1
            status = "low_confidence"
            fmt = cfg.low_confidence_format
        else:
            stats["recognized"] += 1
            status = "recognized"
            fmt = cfg.name_format

        new_name = build_new_name(person, orig_name, fmt)
        current_display_name = orig_name

        if apply:
            try:
                new_path = safe_rename(fpath, new_name)
                current_display_name = new_path.name
                new_name = new_path.name
                media_url = f"/api/media/{new_path.relative_to(PROJECT_ROOT)}?t={now_ts}"
                status = "renamed_now"
                stats["renamed"] += 1
            except Exception as e:
                stats["error"] += 1

        results.append({
            "original": orig_name,
            "current_name": current_display_name,
            "new_name": new_name,
            "url": media_url,
            "person": person,
            "confidence": conf,
            "reason": reason,
            "status": status,
            "bbox": bbox,
        })

    return {"stats": stats, "results": results, "applied": apply}


@app.get("/api/scan/stream")
async def scan_stream(
    request: Request,
    apply: bool = Query(False),
    force: bool = Query(False),
):
    """
    Quét và nhận diện khuôn mặt thời gian thực theo từng tệp (SSE stream).
    Cho phép giao diện hiển thị ngay từng ảnh được phân tích mà không cần chờ đợi.
    """
    async def event_generator():
        cfg = load_config()
        dataset_root = (PROJECT_ROOT / cfg.dataset_root).resolve()
        unclassified_dir = dataset_root / cfg.unclassified_dir

        engine_obj = get_shared_engine()
        cache = get_shared_cache()

        known_faces, index = load_known_faces(cfg, engine_obj, cache, dataset_root, unclassified_dir)

        media_files = sorted([
            f for f in unclassified_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ALL_VALID_EXTENSIONS
        ], key=lambda f: f.name.lower())

        stats = {
            "recognized": 0,
            "low_confidence": 0,
            "unknown": 0,
            "skipped": 0,
            "renamed": 0,
            "error": 0,
            "total": len(media_files),
        }
        known_names = set(known_faces.keys())
        now_ts = int(time.time() * 1000)

        yield f"data: {json.dumps({'type': 'start', 'total': len(media_files), 'apply': apply})}\n\n"
        await asyncio.sleep(0.02)

        for i, fpath in enumerate(media_files, start=1):
            if await request.is_disconnected() or "scan" in ACTIVE_CANCELLATIONS:
                ACTIVE_CANCELLATIONS.discard("scan")
                yield f"data: {json.dumps({'type': 'cancelled', 'message': 'Đã dừng quét theo yêu cầu', 'stats': stats})}\n\n"
                break

            orig_name = fpath.name
            rel = fpath.relative_to(PROJECT_ROOT)
            media_url = f"/api/media/{rel}?t={now_ts}"
            is_already = is_already_processed(orig_name, known_names, cfg.unknown_prefix)

            # Nếu file đã đổi tên trước đó và không quét cưỡng bức, bỏ qua ONNX inference
            if is_already and not force and not apply:
                stats["skipped"] += 1
                predicted_p = orig_name.split("_")[0]
                item_data = {
                    "original": orig_name,
                    "current_name": orig_name,
                    "new_name": orig_name,
                    "url": media_url,
                    "person": predicted_p,
                    "confidence": 100.0,
                    "reason": f"Đã được gắn nhãn '{predicted_p}'",
                    "status": "already_named",
                    "bbox": None,
                }
                yield f"data: {json.dumps({'type': 'item', 'index': i, 'total': len(media_files), 'stats': stats, 'item': item_data})}\n\n"
                await asyncio.sleep(0.01)
                continue

            # Trích xuất bounding box và vector đặc trưng
            img_rgb = load_image_rgb(fpath, max_size=cfg.image_max_size)
            bbox = None
            face_enc = None

            if img_rgb is not None:
                h, w, _ = img_rgb.shape
                faces = engine_obj.app.get(img_rgb[:, :, ::-1]) if engine_obj.engine == "insightface" else []
                if faces:
                    best_face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
                    b = best_face.bbox
                    bbox = {
                        "x": float(b[0] / w),
                        "y": float(b[1] / h),
                        "w": float((b[2] - b[0]) / w),
                        "h": float((b[3] - b[1]) / h),
                        "score": round(float(best_face.det_score) * 100, 1),
                    }
                    emb = best_face.embedding
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        face_enc = (emb / norm).astype(np.float32)

            if is_already and not apply:
                stats["skipped"] += 1
                predicted_p = orig_name.split("_")[0]
                conf = 100.0
                if face_enc is not None:
                    _, _, c_val = index.match_single(face_enc, top_k=cfg.top_k_neighbors)
                    conf = c_val

                item_data = {
                    "original": orig_name,
                    "current_name": orig_name,
                    "new_name": orig_name,
                    "url": media_url,
                    "person": predicted_p,
                    "confidence": conf,
                    "reason": f"Đã được gắn nhãn '{predicted_p}'",
                    "status": "already_named",
                    "bbox": bbox,
                }
                yield f"data: {json.dumps({'type': 'item', 'index': i, 'total': len(media_files), 'stats': stats, 'item': item_data})}\n\n"
                await asyncio.sleep(0.01)
                continue

            if face_enc is None:
                person = cfg.unknown_prefix
                reason = "Không phát hiện khuôn mặt"
                conf = 0.0
            else:
                person, reason, conf = index.match_single(
                    face_enc=face_enc,
                    top_k=cfg.top_k_neighbors,
                    threshold=cfg.cosine_threshold,
                    margin_threshold=cfg.margin_threshold,
                    unknown_prefix=cfg.unknown_prefix,
                )

            is_unknown = (person == cfg.unknown_prefix)
            is_low = (not is_unknown and conf < cfg.low_confidence_threshold)

            if is_unknown:
                stats["unknown"] += 1
                status = "unknown"
                fmt = cfg.name_format
            elif is_low:
                stats["low_confidence"] += 1
                status = "low_confidence"
                fmt = cfg.low_confidence_format
            else:
                stats["recognized"] += 1
                status = "recognized"
                fmt = cfg.name_format

            new_name = build_new_name(person, orig_name, fmt)
            current_display_name = orig_name

            if apply:
                try:
                    new_path = safe_rename(fpath, new_name)
                    current_display_name = new_path.name
                    new_name = new_path.name
                    media_url = f"/api/media/{new_path.relative_to(PROJECT_ROOT)}?t={now_ts}"
                    status = "renamed_now"
                    stats["renamed"] += 1
                except Exception as e:
                    stats["error"] += 1

            item_data = {
                "original": orig_name,
                "current_name": current_display_name,
                "new_name": new_name,
                "url": media_url,
                "person": person,
                "confidence": conf,
                "reason": reason,
                "status": status,
                "bbox": bbox,
            }
            yield f"data: {json.dumps({'type': 'item', 'index': i, 'total': len(media_files), 'stats': stats, 'item': item_data})}\n\n"
            await asyncio.sleep(0.01)

        yield f"data: {json.dumps({'type': 'complete', 'stats': stats, 'applied': apply})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/scan/cancel")
async def cancel_scan():
    ACTIVE_CANCELLATIONS.add("scan")
    return {"success": True, "message": "Đã gửi tín hiệu dừng quét"}


# ────────────────────────────────────────────────────────────
# Clustering & DBSCAN
# ────────────────────────────────────────────────────────────

@app.post("/api/cluster")
async def cluster_unknown_faces(payload: Dict[str, Any] = {}):
    eps = payload.get("eps")
    min_samples = payload.get("min_samples", 2)
    cfg = load_config()
    dataset_root = (PROJECT_ROOT / cfg.dataset_root).resolve()
    unclassified_dir = dataset_root / cfg.unclassified_dir

    engine_obj = get_shared_engine()
    cache = get_shared_cache()

    metric = "cosine" if cfg.engine == "insightface" else "euclidean"
    dbscan_eps = eps if eps is not None else cfg.cluster_eps_cosine

    exclude_dirs = {d.lower() for d in cfg.exclude_dirs}
    exclude_dirs.add(unclassified_dir.name.lower())
    known_person_names = {
        d.name.lower() for d in dataset_root.iterdir()
        if d.is_dir() and d.name.lower() not in exclude_dirs
    }

    candidate_files = []
    for p in unclassified_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in ALL_VALID_EXTENSIONS:
            continue
        n_lower = p.name.lower()
        if n_lower.startswith("low_"):
            continue
        is_known = any(n_lower.startswith(f"{p_name}_") or n_lower.startswith(f"{p_name}-") for p_name in known_person_names)
        if not is_known:
            candidate_files.append(p)

    candidate_files.sort(key=lambda p: p.name.lower())

    encodings_list = []
    valid_file_paths = []

    for fpath in candidate_files:
        cached_enc = cache.get(fpath, cfg.engine, cfg.insightface_model)
        if cached_enc is not None and cached_enc.size > 0:
            encodings_list.append(cached_enc)
            valid_file_paths.append(fpath)
        else:
            embs = engine_obj.extract_from_file(fpath, filter_quality=True, select_single=True)
            if embs:
                enc = embs[0]
                cache.set(fpath, cfg.engine, cfg.insightface_model, enc)
                encodings_list.append(enc)
                valid_file_paths.append(fpath)

    cache.save()

    if not encodings_list:
        return {"clusters": [], "noise": [], "message": "Không có khuôn mặt nào để gom nhóm"}

    from sklearn.cluster import DBSCAN
    X = np.vstack(encodings_list)
    clt = DBSCAN(eps=dbscan_eps, min_samples=min_samples, metric=metric)
    labels = clt.fit_predict(X)

    unique_labels = sorted(list({l for l in labels if l != -1}))

    existing_dirs = {d.name.lower() for d in dataset_root.iterdir() if d.is_dir()}
    next_idx = 1
    while f"{cfg.cluster_prefix.lower()}_{next_idx}" in existing_dirs:
        next_idx += 1

    clusters = []
    for idx_c, cluster_id in enumerate(unique_labels, start=1):
        indices = [i for i, l in enumerate(labels) if l == cluster_id]
        c_files = [valid_file_paths[i] for i in indices]
        c_encs = X[indices]

        if len(c_encs) > 1:
            sim_matrix = np.dot(c_encs, c_encs.T)
            triu = np.triu_indices(len(c_encs), k=1)
            avg_sim = round(float(np.mean(sim_matrix[triu])) * 100, 1)
        else:
            avg_sim = 100.0

        suggested_name = f"{cfg.cluster_prefix}_{next_idx}"
        file_items = []
        for f in c_files:
            rel = f.relative_to(PROJECT_ROOT)
            file_items.append({
                "name": f.name,
                "url": f"/api/media/{rel}",
                "path": str(rel),
            })

        clusters.append({
            "cluster_id": int(cluster_id),
            "suggested_name": suggested_name,
            "count": len(c_files),
            "avg_similarity": avg_sim,
            "files": file_items,
        })
        next_idx += 1

    noise_indices = [i for i, l in enumerate(labels) if l == -1]
    noise_items = []
    for i in noise_indices:
        f = valid_file_paths[i]
        rel = f.relative_to(PROJECT_ROOT)
        noise_items.append({
            "name": f.name,
            "url": f"/api/media/{rel}",
        })

    return {
        "clusters": clusters,
        "noise": noise_items,
        "total_clustered": sum(c["count"] for c in clusters),
        "total_noise": len(noise_items),
    }


@app.post("/api/cluster/apply")
async def apply_cluster_group(payload: Dict[str, Any]):
    cluster_name = payload.get("name")
    files = payload.get("files", [])
    move_to_dataset = payload.get("move_to_dataset", True)

    if not cluster_name or not files:
        raise HTTPException(status_code=400, detail="Thiếu tên nhóm hoặc danh sách tệp")

    cfg = load_config()
    dataset_root = (PROJECT_ROOT / cfg.dataset_root).resolve()
    target_person_dir = dataset_root / cluster_name.lower().strip()
    target_person_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    for fpath_str in files:
        fpath = (PROJECT_ROOT / fpath_str).resolve()
        if fpath.exists() and fpath.is_file():
            # Clean unknown_ prefix if present
            clean_name = fpath.name
            if clean_name.lower().startswith("unknown_"):
                clean_name = clean_name[8:]

            target_filename = f"{cluster_name.lower()}_{clean_name}"
            if move_to_dataset:
                safe_move(fpath, target_person_dir, target_filename=target_filename)
            else:
                safe_rename(fpath, target_filename)
            moved.append(clean_name)

    # Invalidate cache to register new person
    cache = get_shared_cache()
    cache.save()

    return {"success": True, "created_folder": cluster_name, "moved_count": len(moved)}


# ────────────────────────────────────────────────────────────
# Organize, Clean, Prepare Actions
# ────────────────────────────────────────────────────────────

@app.post("/api/organize")
async def api_organize(payload: Dict[str, Any] = {}):
    apply = payload.get("apply", True)
    yes = payload.get("yes", True)
    cfg = load_config()
    run_organize(cfg=cfg, apply=apply, yes=yes, include_low=True)
    return {"success": True, "message": "Đã di chuyển các tệp về đúng thư mục"}


@app.post("/api/clean")
async def api_clean(payload: Dict[str, Any] = {}):
    person = payload.get("person")
    apply = payload.get("apply", True)
    cfg = load_config()
    run_clean(cfg=cfg, target_dir=person, apply=apply)
    return {"success": True, "message": "Đã chuẩn hóa tên tệp thành công"}


@app.post("/api/prepare")
async def api_prepare(payload: Dict[str, Any] = {}):
    count = payload.get("count", 3)
    cfg = load_config()
    run_prepare(cfg=cfg, count=count, copy=True, include_other=True)
    return {"success": True, "message": f"Đã khôi phục và tạo ảnh test với {count} ảnh/người"}


# ────────────────────────────────────────────────────────────
# Real-time Web Downloader Stream (SSE)
# ────────────────────────────────────────────────────────────

@app.post("/api/download/cancel")
async def cancel_download():
    ACTIVE_CANCELLATIONS.add("download")
    return {"success": True, "message": "Đã gửi tín hiệu dừng tiến trình tải"}


@app.get("/api/download/stream")
async def stream_download(
    request: Request,
    query: str = Query(...),
    mode: str = Query("web"),
    person: Optional[str] = Query(None),
    include_videos: bool = Query(True),
    extract_frames: bool = Query(False),
    dest_mode: str = Query("download_person"),
    custom_dir: Optional[str] = Query(None),
    target: Optional[str] = Query(None),
    count: int = Query(10),
    download_all: bool = Query(False),
    same_person: bool = Query(True),
    clean_target: bool = Query(False),
    skip_static_video: bool = Query(True),
):
    async def event_generator():
        cfg = load_config()
        dataset_root = (PROJECT_ROOT / cfg.dataset_root).resolve()
        is_vid = is_video_url(query)
        is_fb = not is_vid and ((mode == "facebook") or ("facebook.com" in query) or ("fb.com" in query))
        is_ig = not is_vid and not is_fb and ((mode == "instagram") or query.strip().startswith("@") or ("instagram.com" in query))
        is_unlimited = download_all or (count <= 0)
        
        # Phân giải thông tin tài khoản hoặc tên người
        clean_user = query.strip()
        real_name = None
        if person and person.strip():
            clean_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", person.strip().lower()).strip("_") or "person"
            clean_user = person.strip()
        elif is_fb:
            clean_user, real_name, user_id, avatar_url = resolve_facebook_profile_info(query)
            clean_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", clean_user.lower()).strip("_") or "fb_face"
        elif is_ig:
            clean_user, real_name, fb_handle = resolve_instagram_profile_info(query)
            clean_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", clean_user.lower()).strip("_") or "face"
        else:
            clean_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", clean_user.lower()).strip("_") or "face"

        # Xác định thư mục đích
        effective_dest = dest_mode
        if target:
            if target == "chua_phan_loai":
                effective_dest = "chua_phan_loai"
            elif target == "download":
                effective_dest = "download_person"
            else:
                effective_dest = "custom"
                custom_dir = target

        if effective_dest == "download_person":
            output_dir = PROJECT_ROOT / "download" / clean_slug
            prefix = clean_slug
        elif effective_dest == "chua_phan_loai":
            output_dir = dataset_root / cfg.unclassified_dir
            prefix = "IMG_TEST"
        elif effective_dest == "dataset_person":
            output_dir = dataset_root / clean_slug
            prefix = clean_slug
        elif effective_dest == "custom" and custom_dir:
            c_path = Path(custom_dir).expanduser()
            if not c_path.is_absolute():
                c_path = (PROJECT_ROOT / c_path).resolve()
            output_dir = c_path
            prefix = clean_slug
        else:
            output_dir = PROJECT_ROOT / "download" / clean_slug
            prefix = clean_slug

        output_dir.mkdir(parents=True, exist_ok=True)
        engine_obj = get_shared_engine()

        try:
            rel_dest = str(output_dir.relative_to(PROJECT_ROOT))
        except Exception:
            rel_dest = str(output_dir)

        active_mode = "video" if is_vid else ("facebook" if is_fb else ("instagram" if is_ig else "web"))
        yield f"data: {json.dumps({'type': 'start', 'query': query, 'mode': active_mode, 'target': rel_dest, 'count': 0 if is_unlimited else count, 'unlimited': is_unlimited})}\n\n"
        await asyncio.sleep(0.04)

        if is_fb and real_name:
            yield f"data: {json.dumps({'type': 'info', 'message': f'✨ Nhận diện danh tính Facebook: {real_name} (@{clean_user})'})}\n\n"
            await asyncio.sleep(0.04)
        elif is_ig and real_name:
            yield f"data: {json.dumps({'type': 'info', 'message': f'✨ Nhận diện danh tính Instagram: {real_name} (@{clean_user})'})}\n\n"
            await asyncio.sleep(0.04)

        # Xóa sạch ảnh cũ nếu người dùng yêu cầu
        if clean_target and output_dir.exists():
            cleaned_c = 0
            for old_f in output_dir.iterdir():
                if old_f.is_file() and old_f.suffix.lower() in ALL_VALID_EXTENSIONS:
                    try:
                        old_f.unlink()
                        cleaned_c += 1
                    except Exception:
                        pass
            if cleaned_c > 0:
                yield f"data: {json.dumps({'type': 'info', 'message': f'🧹 Đã làm sạch {cleaned_c} ảnh cũ trong thư mục {rel_dest} trước khi tải mới.'})}\n\n"
                await asyncio.sleep(0.02)

        anchor_centroid = None
        if same_person:
            anchor_centroid = preload_anchor_from_dir(output_dir, engine_obj)
            if anchor_centroid is not None:
                yield f"data: {json.dumps({'type': 'info', 'message': f'Đã nạp vector mỏ neo từ ảnh có sẵn trong {output_dir.name}!'})}\n\n"

        existing_files = [f.stem for f in output_dir.iterdir() if f.is_file()]
        idx = 1
        clean_prefix = prefix
        while f"{clean_prefix}_{idx:04d}" in existing_files:
            idx += 1

        cand_counter = 0
        saved_count = 0
        seen_urls: Set[str] = set()
        seed_embeddings: List[np.ndarray] = []
        is_cancelled = False

        # 0. Chế độ Video / Highlight / Reel (Cắt frame và nhận diện)
        if is_vid:
            yield f"data: {json.dumps({'type': 'info', 'message': f'🎬 Phát hiện liên kết Video / Highlight. Đang tải video (sử dụng session cookie Brave)...'})}\n\n"
            await asyncio.sleep(0.04)

            temp_vid_dir = Path(tempfile.mkdtemp(prefix="facecls_vids_"))
            try:
                local_vid_path = download_video_stream(query, temp_dir=temp_vid_dir, use_browser_cookies=True)
                if not local_vid_path or not local_vid_path.exists():
                    if "facebook.com/stories/" in query or "fb.com/stories/" in query:
                        err_msg = (
                            "Facebook Stories (dạng link /stories/...) là nội dung tạm thời (24h) của Meta, "
                            "mã hóa qua luồng GraphQL nội bộ trên trình duyệt nên các công cụ tự động (yt-dlp, gallery-dl) không thể bóc tách trực tiếp từ link này. "
                            "Gợi ý: Mở F12 trên trình duyệt (tab Network -> lọc Media hoặc .mp4) để copy link CDN trực tiếp dán vào, "
                            "hoặc dán link Reel / Video bài đăng công khai."
                        )
                    else:
                        err_msg = "Không thể tải video từ URL này. Vui lòng kiểm tra lại liên kết hoặc quyền riêng tư."
                    yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"
                else:
                    sz_kb = local_vid_path.stat().st_size // 1024
                    yield f"data: {json.dumps({'type': 'info', 'message': f'✅ Đã tải xong video ({sz_kb} KB). Đang phân tích chuyển động & trích xuất khuôn mặt...'})}\n\n"
                    await asyncio.sleep(0.04)

                    video_matched = False
                    best_match_sim = 0.0
                    is_static_detected = False

                    # Tự động phát hiện video chỉ có 1 ảnh tĩnh ghép nhạc
                    if skip_static_video:
                        is_static_detected, best_static_frame, static_desc = analyze_video_motion_and_best_frame(local_vid_path, engine_obj=engine_obj)
                        if is_static_detected and best_static_frame is not None:
                            yield f"data: {json.dumps({'type': 'info', 'message': f'📸 Video là 1 ảnh tĩnh ghép nhạc ({static_desc}). Bỏ qua lưu video MP4, chụp 1 frame ảnh sắc nét nhất!'})}\n\n"
                            await asyncio.sleep(0.03)

                            cand_counter += 1
                            h, w = best_static_frame.shape[:2]
                            scale = min(1.0, 320 / max(h, w))
                            preview_small = cv2.resize(cv2.cvtColor(best_static_frame, cv2.COLOR_RGB2BGR), (int(w * scale), int(h * scale)))
                            _, buf = cv2.imencode(".jpg", preview_small, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                            b64_url = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"

                            yield f"data: {json.dumps({'type': 'candidate_found', 'id': cand_counter, 'url': b64_url, 'source': 'video', 'desc': 'Ảnh chụp từ Video tĩnh'})}\n\n"
                            await asyncio.sleep(0.02)

                            embs = engine_obj.extract_face_embeddings(best_static_frame, filter_quality=True, select_single=True)
                            if not embs:
                                yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'no_face', 'reason': 'Không có khuôn mặt'})}\n\n"
                            else:
                                face_enc = embs[0]
                                sim = 1.0
                                is_match = True
                                if same_person:
                                    if anchor_centroid is None:
                                        anchor_centroid = face_enc
                                        yield f"data: {json.dumps({'type': 'anchor_established', 'message': '🎯 Đã xác lập mỏ neo danh tính từ ảnh trích xuất video tĩnh!'})}\n\n"
                                    else:
                                        sim = float(np.dot(anchor_centroid, face_enc))
                                        if sim < 0.38:
                                            is_match = False
                                            yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'rejected', 'sim': round(sim * 100, 1), 'reason': f'Khác người ({sim*100:.1f}%)'})}\n\n"
                                if is_match:
                                    video_matched = True
                                    best_match_sim = sim
                                    filename_frame = f"{clean_prefix}_{idx:04d}.jpg"
                                    save_frame_path = output_dir / filename_frame
                                    while save_frame_path.exists():
                                        idx += 1
                                        filename_frame = f"{clean_prefix}_{idx:04d}.jpg"
                                        save_frame_path = output_dir / filename_frame
                                    img = Image.fromarray(best_static_frame)
                                    img.save(save_frame_path, "JPEG", quality=92)
                                    saved_count += 1
                                    idx += 1
                                    rel_path = save_frame_path.relative_to(PROJECT_ROOT)
                                    url_val = f"/api/media/{rel_path}"
                                    yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'accepted', 'sim': round(sim * 100, 1), 'filename': filename_frame, 'local_url': url_val})}\n\n"
                                    yield f"data: {json.dumps({'type': 'saved', 'is_video': False, 'saved': saved_count, 'total': saved_count, 'unlimited': is_unlimited, 'filename': filename_frame, 'url': url_val, 'sim': round(sim * 100, 1)})}\n\n"
                                    yield f"data: {json.dumps({'type': 'info', 'message': f'✅ Đã chụp và lưu 1 ảnh tĩnh từ video: {filename_frame} (đã bỏ qua file video nặng)'})}\n\n"

                    # Nếu không phải video tĩnh, xử lý lấy mẫu frame và lưu video MP4
                    if not is_static_detected:
                        saved_embeddings: List[np.ndarray] = []
                        max_frame_samples = 40 if is_unlimited else max(20, count * 3)

                        for frame_idx, timestamp_sec, rgb_frame in sample_video_frames(local_vid_path, sample_interval_sec=0.75, max_frames=max_frame_samples):
                            if not is_unlimited and saved_count >= count:
                                break
                            if is_cancelled or "download" in ACTIVE_CANCELLATIONS or await request.is_disconnected():
                                ACTIVE_CANCELLATIONS.discard("download")
                                is_cancelled = True
                                break

                            cand_counter += 1
                            h, w = rgb_frame.shape[:2]
                            scale = min(1.0, 320 / max(h, w))
                            preview_small = cv2.resize(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), (int(w * scale), int(h * scale)))
                            _, buf = cv2.imencode(".jpg", preview_small, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                            b64_url = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"

                            yield f"data: {json.dumps({'type': 'candidate_found', 'id': cand_counter, 'url': b64_url, 'source': 'video', 'desc': f'Frame #{cand_counter} ({timestamp_sec:.1f}s)'})}\n\n"
                            await asyncio.sleep(0.015)

                            embs = engine_obj.extract_face_embeddings(rgb_frame, filter_quality=True, select_single=True)
                            if not embs:
                                yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'no_face', 'reason': 'Không có khuôn mặt'})}\n\n"
                                continue

                            face_enc = embs[0]
                            sim = 1.0

                            if same_person:
                                if anchor_centroid is None:
                                    anchor_centroid = face_enc
                                    yield f"data: {json.dumps({'type': 'anchor_established', 'message': f'🎯 Đã xác lập mỏ neo danh tính từ Frame đầu tiên của video!'})}\n\n"
                                else:
                                    sim = float(np.dot(anchor_centroid, face_enc))
                                    if sim < 0.38:
                                        yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'rejected', 'sim': round(sim * 100, 1), 'reason': f'Khác người ({sim*100:.1f}%)'})}\n\n"
                                        continue

                                    is_dup = any(float(np.dot(pe, face_enc)) > 0.92 for pe in saved_embeddings[-2:])
                                    if not is_dup:
                                        anchor_centroid = 0.95 * anchor_centroid + 0.05 * face_enc
                                        anchor_centroid /= np.linalg.norm(anchor_centroid)

                            video_matched = True
                            best_match_sim = max(best_match_sim, sim)
                            saved_embeddings.append(face_enc)
                            yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'accepted', 'sim': round(sim * 100, 1), 'reason': 'Khớp khuôn mặt'})}\n\n"

                            # Nếu người dùng muốn trích xuất frame ảnh
                            if extract_frames:
                                filename_frame = f"{clean_prefix}_{idx:04d}.jpg"
                                save_frame_path = output_dir / filename_frame
                                while save_frame_path.exists():
                                    idx += 1
                                    filename_frame = f"{clean_prefix}_{idx:04d}.jpg"
                                    save_frame_path = output_dir / filename_frame
                                try:
                                    img = Image.fromarray(rgb_frame)
                                    img.save(save_frame_path, "JPEG", quality=92)
                                    saved_count += 1
                                    idx += 1
                                    rel_path = save_frame_path.relative_to(PROJECT_ROOT)
                                    url_val = f"/api/media/{rel_path}"
                                    yield f"data: {json.dumps({'type': 'saved', 'is_video': False, 'saved': saved_count, 'total': 0 if is_unlimited else count, 'unlimited': is_unlimited, 'filename': filename_frame, 'url': url_val, 'sim': round(sim * 100, 1)})}\n\n"
                                    await asyncio.sleep(0.02)
                                except Exception:
                                    pass

                            # Dừng sớm nếu đã xác thực chắc chắn khuôn mặt trong video
                            if video_matched and not extract_frames and len(saved_embeddings) >= 2:
                                break

                        # Lưu file video gốc (.mp4) trực tiếp vào thư mục đích (chỉ khi là video động)
                        if video_matched and local_vid_path and local_vid_path.exists():
                            ext = local_vid_path.suffix if local_vid_path.suffix else ".mp4"
                            v_filename = f"{clean_prefix}_{idx:04d}{ext}"
                            dest_v_path = output_dir / v_filename
                            while dest_v_path.exists():
                                idx += 1
                                v_filename = f"{clean_prefix}_{idx:04d}{ext}"
                                dest_v_path = output_dir / v_filename
                            try:
                                shutil.copy2(local_vid_path, dest_v_path)
                                saved_count += 1
                                idx += 1
                                rel_v = str(dest_v_path.relative_to(PROJECT_ROOT))
                                url_v = f"/api/media/{rel_v}"
                                yield f"data: {json.dumps({'type': 'saved', 'is_video': True, 'saved': saved_count, 'total': saved_count, 'unlimited': is_unlimited, 'filename': v_filename, 'url': url_v, 'sim': round(best_match_sim * 100, 1)})}\n\n"
                                yield f"data: {json.dumps({'type': 'info', 'message': f'✅ Đã xác thực khuôn mặt & lưu Video thành công: {v_filename}'})}\n\n"
                            except Exception as e:
                                logger.debug(f"Không thể lưu video: {e}")
            finally:
                if temp_vid_dir.exists():
                    shutil.rmtree(temp_vid_dir, ignore_errors=True)

            # Kết thúc luồng video
            if is_cancelled:
                yield f"data: {json.dumps({'type': 'cancelled', 'saved': saved_count, 'message': 'Đã dừng xử lý video theo yêu cầu'})}\n\n"
            else:
                if not video_matched and ("facebook.com/stories/" in query or "fb.com/stories/" in query):
                    msg_finish = "Link Facebook Story yêu cầu link CDN trực tiếp (.mp4) hoặc link Reel công khai."
                else:
                    msg_finish = f"Đã xác thực & tải thành công Video vào {rel_dest}!" if video_matched else f"Không tìm thấy khuôn mặt phù hợp trong Video này."
                yield f"data: {json.dumps({'type': 'complete', 'saved': saved_count, 'total': saved_count, 'target': rel_dest, 'message': msg_finish})}\n\n"
            return

        # 1. Chế độ Mạng Xã Hội (Facebook hoặc Instagram): Cào và stream ngay từng ảnh & avatar
        if is_fb or is_ig:
            platform_name = "Facebook" if is_fb else "Instagram"
            yield f"data: {json.dumps({'type': 'info', 'message': f'Đang kết nối {platform_name} mirror cho tài khoản @{clean_user}...'})}\n\n"
            await asyncio.sleep(0.04)

            max_posts_limit = 0 if is_unlimited else max(40, count * 4)
            img_iter = (
                iter_facebook_images(query, max_count=max_posts_limit)
                if is_fb
                else iter_instagram_images(query, max_count=max_posts_limit, include_videos=include_videos)
            )

            vids_dir = output_dir / "videos"
            saved_vid_embeddings: List[np.ndarray] = []
            reel_idx = 1
            hl_idx = 1

            for item in img_iter:
                cand_url = item[0]
                is_avatar = item[1]
                desc = item[2]
                is_video_post = item[3] if len(item) > 3 else False

                if not is_unlimited and saved_count >= count:
                    break
                if is_cancelled:
                    break
                if "download" in ACTIVE_CANCELLATIONS or await request.is_disconnected():
                    ACTIVE_CANCELLATIONS.discard("download")
                    is_cancelled = True
                    break
                if cand_url in seen_urls:
                    continue
                seen_urls.add(cand_url)

                cand_counter += 1
                cand_source = "video" if is_video_post else ("facebook" if is_fb else "instagram")
                yield f"data: {json.dumps({'type': 'candidate_found', 'id': cand_counter, 'url': cand_url, 'source': cand_source, 'is_avatar': is_avatar, 'desc': desc})}\n\n"
                await asyncio.sleep(0.02)

                if is_video_post:
                    yield f"data: {json.dumps({'type': 'candidate_evaluating', 'id': cand_counter})}\n\n"
                    temp_reel_dir = Path(tempfile.mkdtemp(prefix="facecls_reel_"))
                    reel_matched = False
                    best_reel_sim = 0.0
                    try:
                        local_reel = download_video_stream(cand_url, temp_dir=temp_reel_dir, use_browser_cookies=True)
                        if not local_reel or not local_reel.exists():
                            yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'error', 'reason': 'Không tải được video Reel'})}\n\n"
                            continue

                        # Kiểm tra xem Reel này có phải là 1 ảnh tĩnh ghép nhạc không
                        if skip_static_video:
                            is_static_reel, best_reel_frame, static_desc = analyze_video_motion_and_best_frame(local_reel, engine_obj=engine_obj)
                            if is_static_reel and best_reel_frame is not None:
                                embs = engine_obj.extract_face_embeddings(best_reel_frame, filter_quality=True, select_single=True)
                                if not embs:
                                    yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'no_face', 'reason': 'Không có khuôn mặt'})}\n\n"
                                else:
                                    face_enc = embs[0]
                                    sim = 1.0
                                    is_match = True
                                    if same_person:
                                        if anchor_centroid is None:
                                            anchor_centroid = face_enc
                                            yield f"data: {json.dumps({'type': 'anchor_established', 'message': '🎯 Đã xác lập mỏ neo danh tính từ Reel tĩnh!'})}\n\n"
                                        else:
                                            sim = float(np.dot(anchor_centroid, face_enc))
                                            if sim < 0.38:
                                                is_match = False
                                                yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'rejected', 'sim': round(sim * 100, 1), 'reason': f'Khác người ({sim*100:.1f}%)'})}\n\n"
                                    if is_match:
                                        filename_frame = f"{clean_prefix}_{idx:04d}.jpg"
                                        save_frame_path = output_dir / filename_frame
                                        while save_frame_path.exists():
                                            idx += 1
                                            filename_frame = f"{clean_prefix}_{idx:04d}.jpg"
                                            save_frame_path = output_dir / filename_frame
                                        try:
                                            img = Image.fromarray(best_reel_frame)
                                            img.save(save_frame_path, "JPEG", quality=92)
                                            saved_count += 1
                                            idx += 1
                                            rel_path = save_frame_path.relative_to(PROJECT_ROOT)
                                            url_val = f"/api/media/{rel_path}"
                                            yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'accepted', 'sim': round(sim * 100, 1), 'filename': filename_frame, 'local_url': url_val})}\n\n"
                                            yield f"data: {json.dumps({'type': 'saved', 'is_video': False, 'saved': saved_count, 'total': 0 if is_unlimited else count, 'unlimited': is_unlimited, 'filename': filename_frame, 'url': url_val, 'sim': round(sim * 100, 1)})}\n\n"
                                            yield f"data: {json.dumps({'type': 'info', 'message': f'📸 Reel là 1 ảnh tĩnh ghép nhạc ({static_desc}) -> Đã chụp 1 ảnh JPG thay vì lưu video'})}\n\n"
                                            await asyncio.sleep(0.02)
                                        except Exception:
                                            pass
                                continue

                        for f_idx, t_sec, rgb_frame in sample_video_frames(local_reel, sample_interval_sec=0.75, max_frames=15):
                            if (not is_unlimited and saved_count >= count) or is_cancelled:
                                break
                            if "download" in ACTIVE_CANCELLATIONS or await request.is_disconnected():
                                ACTIVE_CANCELLATIONS.discard("download")
                                is_cancelled = True
                                break

                            embs = engine_obj.extract_face_embeddings(rgb_frame, filter_quality=True, select_single=True)
                            if not embs:
                                continue
                            face_enc = embs[0]
                            sim = 1.0
                            if same_person and anchor_centroid is not None:
                                sim = float(np.dot(anchor_centroid, face_enc))
                                if sim < 0.38:
                                    continue
                                anchor_centroid = 0.95 * anchor_centroid + 0.05 * face_enc
                                anchor_centroid /= np.linalg.norm(anchor_centroid)
                            elif same_person and anchor_centroid is None:
                                anchor_centroid = face_enc

                            reel_matched = True
                            best_reel_sim = max(best_reel_sim, sim)

                            if extract_frames:
                                filename = f"{clean_prefix}_{idx:04d}.jpg"
                                save_path = output_dir / filename
                                while save_path.exists():
                                    idx += 1
                                    filename = f"{clean_prefix}_{idx:04d}.jpg"
                                    save_path = output_dir / filename
                                try:
                                    img = Image.fromarray(rgb_frame)
                                    img.save(save_path, "JPEG", quality=92)
                                    saved_count += 1
                                    idx += 1
                                    rel_path = save_path.relative_to(PROJECT_ROOT)
                                    url_val = f"/api/media/{rel_path}"
                                    yield f"data: {json.dumps({'type': 'saved', 'is_video': False, 'saved': saved_count, 'total': 0 if is_unlimited else count, 'unlimited': is_unlimited, 'filename': filename, 'url': url_val, 'sim': round(sim * 100, 1)})}\n\n"
                                    await asyncio.sleep(0.02)
                                except Exception:
                                    pass

                            if reel_matched and not extract_frames:
                                break

                        if reel_matched and local_reel and local_reel.exists():
                            ext = local_reel.suffix if local_reel.suffix else ".mp4"
                            dest_reel_name = f"{clean_prefix}_{idx:04d}{ext}"
                            save_vid_path = output_dir / dest_reel_name
                            while save_vid_path.exists():
                                idx += 1
                                dest_reel_name = f"{clean_prefix}_{idx:04d}{ext}"
                                save_vid_path = output_dir / dest_reel_name
                            try:
                                shutil.copy2(local_reel, save_vid_path)
                                saved_count += 1
                                idx += 1
                                rel_v = str(save_vid_path.relative_to(PROJECT_ROOT))
                                url_v = f"/api/media/{rel_v}"
                                yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'accepted', 'sim': round(best_reel_sim * 100, 1), 'filename': dest_reel_name, 'local_url': url_v})}\n\n"
                                yield f"data: {json.dumps({'type': 'saved', 'is_video': True, 'saved': saved_count, 'total': 0 if is_unlimited else count, 'unlimited': is_unlimited, 'filename': dest_reel_name, 'url': url_v, 'sim': round(best_reel_sim * 100, 1)})}\n\n"
                                await asyncio.sleep(0.03)
                            except Exception:
                                pass
                        else:
                            yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'rejected', 'reason': 'Không khớp người hoặc không rõ mặt'})}\n\n"
                    finally:
                        if temp_reel_dir.exists():
                            shutil.rmtree(temp_reel_dir, ignore_errors=True)
                    continue

                yield f"data: {json.dumps({'type': 'candidate_evaluating', 'id': cand_counter})}\n\n"
                img = download_and_verify_image(cand_url, min_size=100 if is_avatar else 160, timeout=5)
                if img is None:
                    yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'error', 'reason': 'Không tải được ảnh'})}\n\n"
                    continue

                arr = np.array(img)
                embs = engine_obj.extract_face_embeddings(arr, filter_quality=True, select_single=True)
                if not embs:
                    yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'no_face', 'reason': 'Không có khuôn mặt'})}\n\n"
                    continue

                face_enc = embs[0]
                sim = 1.0

                if same_person:
                    if is_avatar and anchor_centroid is None:
                        anchor_centroid = face_enc
                        yield f"data: {json.dumps({'type': 'anchor_established', 'message': f'🎯 Đã xác lập mỏ neo danh tính từ Avatar {platform_name} chính chủ của {real_name or clean_user}!'})}\n\n"
                    elif anchor_centroid is None:
                        seed_embeddings.append(face_enc)
                        if len(seed_embeddings) >= min(3, 5):
                            s_mat = np.array(seed_embeddings)
                            sims = s_mat @ s_mat.T
                            avg_s = np.mean(sims, axis=1)
                            c = np.mean(s_mat[avg_s >= np.median(avg_s)], axis=0)
                            anchor_centroid = c / np.linalg.norm(c)
                            yield f"data: {json.dumps({'type': 'anchor_established', 'message': f'🎯 Đã xác lập mỏ neo danh tính cho {real_name or clean_user}!'})}\n\n"
                    else:
                        sim = float(np.dot(anchor_centroid, face_enc))
                        if sim < 0.38:
                            yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'rejected', 'sim': round(sim * 100, 1), 'reason': f'Khác người ({sim*100:.1f}%)'})}\n\n"
                            yield f"data: {json.dumps({'type': 'skipped', 'message': f'Bỏ qua (Khác người, sim={sim*100:.1f}%)'})}\n\n"
                            await asyncio.sleep(0.02)
                            continue
                        anchor_centroid = 0.95 * anchor_centroid + 0.05 * face_enc
                        anchor_centroid /= np.linalg.norm(anchor_centroid)

                filename = f"{clean_prefix}_{idx:04d}.jpg"
                save_path = output_dir / filename
                while save_path.exists():
                    idx += 1
                    filename = f"{clean_prefix}_{idx:04d}.jpg"
                    save_path = output_dir / filename

                try:
                    img.save(save_path, "JPEG", quality=92)
                    saved_count += 1
                    idx += 1
                    try:
                        rel_path = save_path.relative_to(PROJECT_ROOT)
                        url_val = f"/api/media/{rel_path}"
                    except Exception:
                        url_val = ""

                    yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'accepted', 'sim': round(sim * 100, 1), 'filename': filename, 'local_url': url_val})}\n\n"
                    msg_data = {
                        "type": "saved",
                        "saved": saved_count,
                        "total": 0 if is_unlimited else count,
                        "unlimited": is_unlimited,
                        "filename": filename,
                        "url": url_val,
                        "sim": round(sim * 100, 1),
                    }
                    yield f"data: {json.dumps(msg_data)}\n\n"
                    await asyncio.sleep(0.04)
                except Exception:
                    pass

            # 1.1 Nếu là Instagram và bật Kèm Video/Highlights: cào toàn bộ Highlights qua gallery-dl & session Brave
            if is_ig and include_videos and not is_cancelled and (is_unlimited or saved_count < count):
                yield f"data: {json.dumps({'type': 'info', 'message': f'🎬 Đang quét bộ sưu tập Highlights của @{clean_user} qua phiên đăng nhập Brave...'})}\n\n"
                await asyncio.sleep(0.04)

                saved_vid_embeddings: List[np.ndarray] = []
                temp_hl_dir = Path(tempfile.mkdtemp(prefix="facecls_ig_hl_"))
                try:
                    for hl_url, is_video_hl, hl_title in iter_instagram_highlights(query):
                        if (not is_unlimited and saved_count >= count) or is_cancelled:
                            break
                        if "download" in ACTIVE_CANCELLATIONS or await request.is_disconnected():
                            ACTIVE_CANCELLATIONS.discard("download")
                            is_cancelled = True
                            break
                        if hl_url in seen_urls:
                            continue
                        seen_urls.add(hl_url)

                        cand_counter += 1
                        if is_video_hl:
                            yield f"data: {json.dumps({'type': 'candidate_found', 'id': cand_counter, 'url': hl_url, 'source': 'video', 'desc': hl_title})}\n\n"
                            yield f"data: {json.dumps({'type': 'candidate_evaluating', 'id': cand_counter})}\n\n"
                            await asyncio.sleep(0.02)

                            local_hl_vid = download_video_stream(hl_url, temp_dir=temp_hl_dir, use_browser_cookies=True)
                            if not local_hl_vid or not local_hl_vid.exists():
                                yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'error', 'reason': 'Không tải được video highlight'})}\n\n"
                                continue

                            # Kiểm tra xem Highlight này có phải là 1 ảnh tĩnh ghép nhạc không
                            if skip_static_video:
                                is_static_hl, best_hl_frame, static_desc = analyze_video_motion_and_best_frame(local_hl_vid, engine_obj=engine_obj)
                                if is_static_hl and best_hl_frame is not None:
                                    embs = engine_obj.extract_face_embeddings(best_hl_frame, filter_quality=True, select_single=True)
                                    if not embs:
                                        yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'no_face', 'reason': 'Không có khuôn mặt'})}\n\n"
                                    else:
                                        face_enc = embs[0]
                                        sim = 1.0
                                        is_match = True
                                        if same_person:
                                            if anchor_centroid is None:
                                                anchor_centroid = face_enc
                                                yield f"data: {json.dumps({'type': 'anchor_established', 'message': '🎯 Đã xác lập mỏ neo danh tính từ Highlight tĩnh!'})}\n\n"
                                            else:
                                                sim = float(np.dot(anchor_centroid, face_enc))
                                                if sim < 0.38:
                                                    is_match = False
                                                    yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'rejected', 'sim': round(sim * 100, 1), 'reason': f'Khác người ({sim*100:.1f}%)'})}\n\n"
                                        if is_match:
                                            filename_frame = f"{clean_prefix}_{idx:04d}.jpg"
                                            save_frame_path = output_dir / filename_frame
                                            while save_frame_path.exists():
                                                idx += 1
                                                filename_frame = f"{clean_prefix}_{idx:04d}.jpg"
                                                save_frame_path = output_dir / filename_frame
                                            try:
                                                img = Image.fromarray(best_hl_frame)
                                                img.save(save_frame_path, "JPEG", quality=92)
                                                saved_count += 1
                                                idx += 1
                                                rel_path = save_frame_path.relative_to(PROJECT_ROOT)
                                                url_val = f"/api/media/{rel_path}"
                                                yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'accepted', 'sim': round(sim * 100, 1), 'filename': filename_frame, 'local_url': url_val})}\n\n"
                                                yield f"data: {json.dumps({'type': 'saved', 'is_video': False, 'saved': saved_count, 'total': 0 if is_unlimited else count, 'unlimited': is_unlimited, 'filename': filename_frame, 'url': url_val, 'sim': round(sim * 100, 1)})}\n\n"
                                                yield f"data: {json.dumps({'type': 'info', 'message': f'📸 Highlight là 1 ảnh tĩnh ghép nhạc ({static_desc}) -> Đã chụp 1 ảnh JPG thay vì lưu video'})}\n\n"
                                                await asyncio.sleep(0.02)
                                            except Exception:
                                                pass
                                    continue

                            hl_matched = False
                            best_hl_sim = 0.0
                            # Lấy mẫu frame từ video highlight
                            for f_idx, t_sec, rgb_frame in sample_video_frames(local_hl_vid, sample_interval_sec=0.75, max_frames=15):
                                if (not is_unlimited and saved_count >= count) or is_cancelled:
                                    break
                                if "download" in ACTIVE_CANCELLATIONS or await request.is_disconnected():
                                    ACTIVE_CANCELLATIONS.discard("download")
                                    is_cancelled = True
                                    break

                                embs = engine_obj.extract_face_embeddings(rgb_frame, filter_quality=True, select_single=True)
                                if not embs:
                                    continue
                                face_enc = embs[0]
                                sim = 1.0
                                if same_person and anchor_centroid is not None:
                                    sim = float(np.dot(anchor_centroid, face_enc))
                                    if sim < 0.38:
                                        continue
                                    anchor_centroid = 0.95 * anchor_centroid + 0.05 * face_enc
                                    anchor_centroid /= np.linalg.norm(anchor_centroid)
                                elif same_person and anchor_centroid is None:
                                    anchor_centroid = face_enc

                                hl_matched = True
                                best_hl_sim = max(best_hl_sim, sim)

                                if extract_frames:
                                    filename = f"{clean_prefix}_{idx:04d}.jpg"
                                    save_path = output_dir / filename
                                    while save_path.exists():
                                        idx += 1
                                        filename = f"{clean_prefix}_{idx:04d}.jpg"
                                        save_path = output_dir / filename
                                    try:
                                        img = Image.fromarray(rgb_frame)
                                        img.save(save_path, "JPEG", quality=92)
                                        saved_count += 1
                                        idx += 1
                                        rel_path = save_path.relative_to(PROJECT_ROOT)
                                        url_val = f"/api/media/{rel_path}"
                                        yield f"data: {json.dumps({'type': 'saved', 'is_video': False, 'saved': saved_count, 'total': 0 if is_unlimited else count, 'unlimited': is_unlimited, 'filename': filename, 'url': url_val, 'sim': round(sim * 100, 1)})}\n\n"
                                        await asyncio.sleep(0.02)
                                    except Exception:
                                        pass

                                if hl_matched and not extract_frames:
                                    break

                            if hl_matched and local_hl_vid and local_hl_vid.exists():
                                ext = local_hl_vid.suffix if local_hl_vid.suffix else ".mp4"
                                dest_hl_name = f"{clean_prefix}_{idx:04d}{ext}"
                                save_vid_path = output_dir / dest_hl_name
                                while save_vid_path.exists():
                                    idx += 1
                                    dest_hl_name = f"{clean_prefix}_{idx:04d}{ext}"
                                    save_vid_path = output_dir / dest_hl_name
                                try:
                                    shutil.copy2(local_hl_vid, save_vid_path)
                                    saved_count += 1
                                    idx += 1
                                    rel_v = str(save_vid_path.relative_to(PROJECT_ROOT))
                                    url_v = f"/api/media/{rel_v}"
                                    yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'accepted', 'sim': round(best_hl_sim * 100, 1), 'filename': dest_hl_name, 'local_url': url_v})}\n\n"
                                    yield f"data: {json.dumps({'type': 'saved', 'is_video': True, 'saved': saved_count, 'total': 0 if is_unlimited else count, 'unlimited': is_unlimited, 'filename': dest_hl_name, 'url': url_v, 'sim': round(best_hl_sim * 100, 1)})}\n\n"
                                    await asyncio.sleep(0.03)
                                except Exception:
                                    pass
                            else:
                                yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'rejected', 'reason': 'Không khớp người hoặc không rõ mặt'})}\n\n"

                            # Xóa file video sau khi đã cắt frame
                            try:
                                if local_hl_vid and local_hl_vid.exists():
                                    local_hl_vid.unlink()
                            except Exception:
                                pass
                        else:
                            # Highlight dạng ảnh tĩnh
                            yield f"data: {json.dumps({'type': 'candidate_found', 'id': cand_counter, 'url': hl_url, 'source': 'instagram', 'desc': hl_title})}\n\n"
                            yield f"data: {json.dumps({'type': 'candidate_evaluating', 'id': cand_counter})}\n\n"
                            img = download_and_verify_image(hl_url, timeout=5)
                            if img is None:
                                yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'error', 'reason': 'Lỗi tải ảnh highlight'})}\n\n"
                                continue
                            arr = np.array(img)
                            embs = engine_obj.extract_face_embeddings(arr, filter_quality=True, select_single=True)
                            if not embs:
                                yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'no_face', 'reason': 'Không có khuôn mặt'})}\n\n"
                                continue
                            face_enc = embs[0]
                            sim = 1.0
                            if same_person and anchor_centroid is not None:
                                sim = float(np.dot(anchor_centroid, face_enc))
                                if sim < 0.38:
                                    yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'rejected', 'sim': round(sim * 100, 1), 'reason': f'Khác người ({sim*100:.1f}%)'})}\n\n"
                                    continue
                                anchor_centroid = 0.95 * anchor_centroid + 0.05 * face_enc
                                anchor_centroid /= np.linalg.norm(anchor_centroid)

                            filename = f"{clean_prefix}_{idx:04d}.jpg"
                            save_path = output_dir / filename
                            while save_path.exists():
                                idx += 1
                                filename = f"{clean_prefix}_{idx:04d}.jpg"
                                save_path = output_dir / filename
                            try:
                                img.save(save_path, "JPEG", quality=92)
                                saved_count += 1
                                idx += 1
                                try:
                                    rel_path = save_path.relative_to(PROJECT_ROOT)
                                    url_val = f"/api/media/{rel_path}"
                                except Exception:
                                    url_val = ""
                                yield f"data: {json.dumps({'type': 'candidate_status', 'id': cand_counter, 'status': 'accepted', 'sim': round(sim * 100, 1), 'filename': filename, 'local_url': url_val})}\n\n"
                                yield f"data: {json.dumps({'type': 'saved', 'saved': saved_count, 'total': 0 if is_unlimited else count, 'unlimited': is_unlimited, 'filename': filename, 'url': url_val, 'sim': round(sim * 100, 1)})}\n\n"
                                await asyncio.sleep(0.03)
                            except Exception:
                                pass
                finally:
                    if temp_hl_dir.exists():
                        shutil.rmtree(temp_hl_dir, ignore_errors=True)

        # 2. Chế độ Web / Bổ sung nếu chưa đủ target count và KHÔNG phải Facebook/Instagram đã có bài đăng
        if not ((is_fb or is_ig) and (is_unlimited or saved_count > 0)):
            if (is_fb or is_ig) and real_name:
                search_queries = [
                    f'"{real_name}" chân dung',
                    f'"{real_name}" portrait',
                    f'"{real_name}" {clean_user}',
                ]
            elif is_fb:
                search_queries = [
                    f'"{clean_user}" site:facebook.com',
                ]
            elif is_ig:
                search_queries = [
                    f'"{clean_user}" site:instagram.com',
                ]
            else:
                search_queries = [
                    query,
                    f"{query} portrait",
                    f"{query} face photo",
                    f"{query} interview",
                ]

            for q in search_queries:
                if (not is_unlimited and saved_count >= count) or is_cancelled:
                    break
                if "download" in ACTIVE_CANCELLATIONS or await request.is_disconnected():
                    ACTIVE_CANCELLATIONS.discard("download")
                    is_cancelled = True
                    break

                first = 0
                while first <= 105 and (is_unlimited or saved_count < count):
                    if "download" in ACTIVE_CANCELLATIONS or await request.is_disconnected():
                        ACTIVE_CANCELLATIONS.discard("download")
                        is_cancelled = True
                        break

                    yield f"data: {json.dumps({'type': 'searching', 'message': f'Đang tìm kiếm ảnh...'})}\n\n"
                    await asyncio.sleep(0.05)

                    new_urls = search_image_urls_page(q, first=first, count=35)
                    if not new_urls:
                        first += 35
                        continue

                    urls_to_process = [u for u in new_urls if u not in seen_urls]
                    for u in urls_to_process:
                        seen_urls.add(u)
                        cand_counter += 1
                        yield f"data: {json.dumps({'type': 'candidate_found', 'id': cand_counter, 'url': u, 'source': 'web'})}\n\n"
                        await asyncio.sleep(0.005)

                    first += 35

                    for u_idx, url in enumerate(urls_to_process, start=1):
                        current_cand_id = cand_counter - len(urls_to_process) + u_idx
                        if (not is_unlimited and saved_count >= count) or is_cancelled:
                            break
                        if "download" in ACTIVE_CANCELLATIONS or await request.is_disconnected():
                            ACTIVE_CANCELLATIONS.discard("download")
                            is_cancelled = True
                            break

                        yield f"data: {json.dumps({'type': 'candidate_evaluating', 'id': current_cand_id})}\n\n"
                        img = download_and_verify_image(url, timeout=4)
                        if img is None:
                            yield f"data: {json.dumps({'type': 'candidate_status', 'id': current_cand_id, 'status': 'error', 'reason': 'Không tải được ảnh'})}\n\n"
                            continue

                        arr = np.array(img)
                        embs = engine_obj.extract_face_embeddings(arr, filter_quality=True, select_single=True)
                        if not embs:
                            yield f"data: {json.dumps({'type': 'candidate_status', 'id': current_cand_id, 'status': 'no_face', 'reason': 'Không có khuôn mặt'})}\n\n"
                            continue

                        face_enc = embs[0]
                        sim = 1.0

                        if same_person:
                            if anchor_centroid is None:
                                seed_embeddings.append(face_enc)
                                if len(seed_embeddings) >= 5:
                                    s_mat = np.array(seed_embeddings)
                                    sims = s_mat @ s_mat.T
                                    avg_s = np.mean(sims, axis=1)
                                    core_mask = avg_s >= np.median(avg_s)
                                    c = np.mean(s_mat[core_mask], axis=0)
                                    anchor_centroid = c / np.linalg.norm(c)
                                    yield f"data: {json.dumps({'type': 'anchor_established', 'message': f'🎯 Đã xác lập mỏ neo danh tính chuẩn cho {real_name or query}!'})}\n\n"
                            else:
                                sim = float(np.dot(anchor_centroid, face_enc))
                                if sim < 0.38:
                                    yield f"data: {json.dumps({'type': 'candidate_status', 'id': current_cand_id, 'status': 'rejected', 'sim': round(sim * 100, 1), 'reason': f'Khác người ({sim*100:.1f}%)'})}\n\n"
                                    yield f"data: {json.dumps({'type': 'skipped', 'message': f'Bỏ qua (Khác người, sim={sim*100:.1f}%)'})}\n\n"
                                    await asyncio.sleep(0.05)
                                    continue
                                anchor_centroid = 0.95 * anchor_centroid + 0.05 * face_enc
                                anchor_centroid /= np.linalg.norm(anchor_centroid)

                        filename = f"{clean_prefix}_{idx:04d}.jpg"
                        save_path = output_dir / filename
                        while save_path.exists():
                            idx += 1
                            filename = f"{clean_prefix}_{idx:04d}.jpg"
                            save_path = output_dir / filename

                        try:
                            img.save(save_path, "JPEG", quality=92)
                            saved_count += 1
                            idx += 1
                            try:
                                rel_path = save_path.relative_to(PROJECT_ROOT)
                                url_val = f"/api/media/{rel_path}"
                            except Exception:
                                url_val = ""

                            yield f"data: {json.dumps({'type': 'candidate_status', 'id': current_cand_id, 'status': 'accepted', 'sim': round(sim * 100, 1), 'filename': filename, 'local_url': url_val})}\n\n"
                            msg_data = {
                                "type": "saved",
                                "saved": saved_count,
                                "total": 0 if is_unlimited else count,
                                "unlimited": is_unlimited,
                                "filename": filename,
                                "url": url_val,
                                "sim": round(sim * 100, 1),
                            }
                            yield f"data: {json.dumps(msg_data)}\n\n"
                            await asyncio.sleep(0.08)
                        except Exception:
                            pass

        if is_cancelled:
            yield f"data: {json.dumps({'type': 'cancelled', 'message': 'Đã dừng tiến trình tải theo yêu cầu', 'saved': saved_count, 'target': rel_dest})}\n\n"
        elif saved_count == 0:
            yield f"data: {json.dumps({'type': 'no_results', 'message': f'⚠️ Không tìm thấy ảnh chân dung công khai cho {query}. Tài khoản có thể đang ở chế độ Riêng tư (Private) hoặc không có bài viết công khai.', 'target': rel_dest})}\n\n"
        else:
            finish_msg = (
                f"Đã quét hết toàn bộ bài viết trên Instagram và lưu thành công {saved_count} ảnh vào {rel_dest}!"
                if (is_ig and is_unlimited)
                else f"Đã thu thập xong {saved_count} ảnh vào {rel_dest}!"
            )
            yield f"data: {json.dumps({'type': 'complete', 'saved': saved_count, 'total': saved_count if is_unlimited else count, 'unlimited': is_unlimited, 'message': finish_msg, 'target': rel_dest})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ────────────────────────────────────────────────────────────
# Single Face Playground (Interactive Inspection)
# ────────────────────────────────────────────────────────────

@app.post("/api/playground")
async def analyze_single_face(file: UploadFile = File(...)):
    """Kiểm tra và so khớp 1 ảnh đơn lẻ với toàn bộ database."""
    content = await file.read()
    try:
        from io import BytesIO
        pil_img = Image.open(BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được tệp ảnh")

    cfg = load_config()
    dataset_root = (PROJECT_ROOT / cfg.dataset_root).resolve()
    engine_obj = get_shared_engine()
    cache = get_shared_cache()

    known_faces, index = load_known_faces(cfg, engine_obj, cache, dataset_root)

    arr = np.array(pil_img)
    h, w, _ = arr.shape
    faces = engine_obj.app.get(arr[:, :, ::-1]) if engine_obj.engine == "insightface" else []

    if not faces:
        return {"faces_found": 0, "results": []}

    face_results = []
    for f in faces:
        b = f.bbox
        bbox = {
            "x": float(b[0] / w),
            "y": float(b[1] / h),
            "w": float((b[2] - b[0]) / w),
            "h": float((b[3] - b[1]) / h),
            "score": round(float(f.det_score) * 100, 1),
        }
        emb = f.embedding
        norm = np.linalg.norm(emb)
        if norm == 0:
            continue
        face_enc = (emb / norm).astype(np.float32)

        # Tính toán top 3 người giống nhất
        sims = index.matrix @ face_enc
        scores = {}
        for person, s_slice in index.person_slices.items():
            p_sims = sims[s_slice]
            if len(p_sims) > 0:
                k = min(3, len(p_sims))
                top_k_vals = np.partition(p_sims, -k)[-k:]
                scores[person] = float(np.mean(top_k_vals))

        sorted_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_candidates = []
        for name, score in sorted_matches:
            # Avatar of candidate
            avatar_url = None
            p_dir = dataset_root / name
            if p_dir.exists():
                imgs = [p for p in p_dir.iterdir() if p.is_file() and p.suffix.lower() in ALL_VALID_EXTENSIONS]
                if imgs:
                    avatar_url = f"/api/media/{imgs[0].relative_to(PROJECT_ROOT)}"

            top_candidates.append({
                "person": name,
                "similarity": round(score * 100, 1),
                "avatar_url": avatar_url,
            })

        best_person, reason, conf = index.match_single(
            face_enc=face_enc,
            top_k=cfg.top_k_neighbors,
            threshold=cfg.cosine_threshold,
            margin_threshold=cfg.margin_threshold,
        )

        face_results.append({
            "bbox": bbox,
            "best_match": best_person,
            "confidence": conf,
            "reason": reason,
            "top_candidates": top_candidates,
        })

    return {
        "faces_found": len(faces),
        "results": face_results,
    }


# ────────────────────────────────────────────────────────────
# Settings & Config API
# ────────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    cfg = load_config()
    return cfg.to_dict()


@app.post("/api/config")
async def update_config(payload: Dict[str, Any]):
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    for k, v in payload.items():
        if k in data or hasattr(AppConfig, k):
            data[k] = v

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    # Invalidate cached engine so new config takes effect
    global _CACHED_ENGINE, _CACHED_INDEX
    _CACHED_ENGINE = None
    _CACHED_INDEX = None

    return {"success": True, "config": data}


@app.post("/api/cache/clear")
async def clear_cache():
    cache = get_shared_cache()
    cache.clear()
    return {"success": True, "message": "Đã xóa toàn bộ cache embedding"}
