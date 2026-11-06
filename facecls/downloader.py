"""Module tự động tìm kiếm và tải ảnh khuôn mặt từ web với cơ chế phân trang liên tục
và xác thực danh tính đồng nhất (Same-Person Identity Verification)."""

import re
import urllib.parse
import urllib.request
import logging
import html
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set
from io import BytesIO
import click
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("face_renamer")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
        "Gecko/20100101 Firefox/120.0"
    )
}


def search_image_urls_page(query: str, first: int = 0, count: int = 35) -> List[str]:
    """Tìm nạp danh sách URL ảnh từ một trang tìm kiếm của Bing."""
    encoded_query = urllib.parse.quote(query)
    endpoint = (
        f"https://www.bing.com/images/async?"
        f"q={encoded_query}&first={first}&count={count}&scenario=ImageBasicHover"
    )
    req = urllib.request.Request(endpoint, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            html = res.read().decode("utf-8", errors="ignore")
            found = re.findall(r'murl&quot;:&quot;(https?://[^&"\s]+)&quot;', html)
            return found or []
    except Exception as e:
        logger.debug(f"Lỗi phân trang Bing first={first}: {e}")
        return []


def resolve_instagram_profile_info(raw_input: str) -> tuple[str, Optional[str], Optional[str]]:
    """
    Trích xuất username, tên thật và tài khoản mạng xã hội liên kết từ đường link hoặc handle Instagram.
    Ví dụ: 'https://www.instagram.com/lhtm080104/' -> ('lhtm080104', 'La Hoàng Thảo My', 'lahoangthaomy04')
    """
    cleaned = raw_input.strip()
    if "instagram.com/" in cleaned:
        cleaned = cleaned.split("instagram.com/")[-1].strip("/").split("?")[0]
    username = cleaned.lstrip("@").lower()

    real_name = None
    fb_handle = None
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        q = urllib.parse.quote(f"{username} instagram")
        url = f"https://html.duckduckgo.com/html/?q={q}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            m_name = re.search(r"([A-ZÀ-Ỹa-zà-ỹ\s]{3,35})\s*\(@?" + re.escape(username) + r"\)", html)
            if m_name:
                real_name = m_name.group(1).strip()
            m_fb = re.search(r"facebook\.com/([a-zA-Z0-9_\.]+)", html)
            if m_fb:
                fb_handle = m_fb.group(1).strip()
    except Exception as e:
        logger.debug(f"Không thể tra cứu thông tin Instagram cho @{username}: {e}")

    return username, real_name, fb_handle


def iter_instagram_images(username: str, max_count: Optional[int] = 40, include_videos: bool = False):
    """
    Generator tìm nạp từng URL ảnh & video thật công khai từ tài khoản Instagram (@username hoặc link profile).
    Hỗ trợ tải tất cả bài viết cho đến khi hết ảnh trên profile nếu max_count <= 0 hoặc None.
    Hỗ trợ cào trọn vẹn cả các album nhiều ảnh (Carousel / Sidecar) và video Reels.
    Yields (url, is_avatar, caption, is_video) lần lượt ngay khi phát hiện để hiển thị trực tiếp lên UI.
    """
    clean_user, real_name, fb_handle = resolve_instagram_profile_info(username)
    yielded = set()
    unlimited = (max_count is None or max_count <= 0)

    # 1. Trực tiếp lấy avatar và bài đăng chính chủ bằng Instaloader
    try:
        import instaloader
        L = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            quiet=True,
        )
        L.context._rate_controller.sleep = lambda *args, **kwargs: None
        L.context._rate_controller.wait_before_query = lambda *args, **kwargs: None

        profile = instaloader.Profile.from_username(L.context, clean_user)
        if profile.full_name and not real_name:
            real_name = profile.full_name

        # Ưu tiên phát avatar chính chủ làm mỏ neo đầu tiên
        pic_url = getattr(profile, "profile_pic_url_hd", None) or getattr(profile, "profile_pic_url", None)
        if pic_url and pic_url not in yielded:
            yielded.add(pic_url)
            yield (pic_url, True, f"Avatar Instagram của {real_name or clean_user}", False)

        for post in profile.get_posts():
            if not unlimited and len(yielded) >= max_count:
                break
            caption = (post.caption[:50] + "...") if post.caption else "Bài viết Instagram"

            if post.typename == "GraphSidecar":
                # Album nhiều ảnh (Carousel) -> lấy toàn bộ ảnh thành phần
                try:
                    for s_node in post.get_sidecar_nodes():
                        if include_videos and getattr(s_node, "is_video", False) and getattr(s_node, "video_url", None):
                            v_u = s_node.video_url
                            if v_u not in yielded:
                                yielded.add(v_u)
                                yield (v_u, False, f"{caption} (Album Video)", True)
                                if not unlimited and len(yielded) >= max_count:
                                    break
                        s_url = getattr(s_node, "display_url", None)
                        if s_url and s_url not in yielded:
                            yielded.add(s_url)
                            yield (s_url, False, f"{caption} (Album)", False)
                            if not unlimited and len(yielded) >= max_count:
                                break
                except Exception:
                    if post.url and post.url not in yielded:
                        yielded.add(post.url)
                        yield (post.url, False, caption, False)
            elif post.is_video:
                if include_videos and getattr(post, "video_url", None):
                    v_u = post.video_url
                    if v_u not in yielded:
                        yielded.add(v_u)
                        yield (v_u, False, f"Video Reel: {caption}", True)
                if post.url and post.url not in yielded:
                    yielded.add(post.url)
                    yield (post.url, False, f"Ảnh bìa Video: {caption}", False)
            else:
                if post.url and post.url not in yielded:
                    yielded.add(post.url)
                    yield (post.url, False, caption, False)
    except Exception as e:
        logger.debug(f"Instaloader không thể truy xuất trực tiếp @{clean_user}: {e}")

    # 2. Fallback tìm kiếm ảnh công khai có định hướng khi tài khoản Instagram bị hạn chế / private
    if not yielded:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        fallback_cap = 60 if unlimited else max_count
        target_queries = []
        if real_name:
            target_queries.append(f'"{real_name}" chân dung')
            target_queries.append(f'"{real_name}" portrait')
            target_queries.append(f'"{real_name}" {clean_user}')
        if fb_handle:
            target_queries.append(f'"{fb_handle}" facebook photo')

        target_queries.append(f'"{clean_user}" site:instagram.com')

        for q_str in target_queries:
            if len(yielded) >= fallback_cap:
                break
            try:
                encoded_query = urllib.parse.quote(q_str)
                bing_url = f"https://www.bing.com/images/async?q={encoded_query}&count=35"
                req = urllib.request.Request(bing_url, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as res:
                    html = res.read().decode("utf-8", errors="ignore")
                    bing_imgs = re.findall(r'murl&quot;:&quot;(https?://[^&"\s]+)&quot;', html)
                    for b_u in bing_imgs:
                        if b_u not in yielded and b_u.startswith("http"):
                            yielded.add(b_u)
                            yield (b_u, False, f"Web: {q_str}")
                            if len(yielded) >= fallback_cap:
                                break
            except Exception as e:
                logger.debug(f"Lỗi tìm kiếm Bing cho query '{q_str}': {e}")


def search_instagram_images(username: str, count: int = 30) -> List[str]:
    """Tìm nạp danh sách URL ảnh thật công khai từ tài khoản Instagram (@username hoặc link profile)."""
    max_c = 0 if count <= 0 else count * 2
    return [item[0] for item in iter_instagram_images(username, max_count=max_c)]


def iter_instagram_highlights(username_or_url: str):
    """
    Tìm nạp các câu chuyện / highlight (video MP4 hoặc ảnh) của tài khoản Instagram qua gallery-dl.
    Sử dụng session cookie từ trình duyệt Brave để truy cập cả tài khoản private / follow.
    Yields (url, is_video, title).
    """
    clean_user, _, _ = resolve_instagram_profile_info(username_or_url)
    target_url = f"https://www.instagram.com/{clean_user}/highlights/"
    cmd = [sys.executable, "-m", "gallery_dl", "--cookies-from-browser", "brave", "-g", target_url]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        yielded_urls = set()
        try:
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                clean_url = line.lstrip("| ").strip()
                if not clean_url.startswith("http"):
                    continue
                if clean_url in yielded_urls:
                    continue
                yielded_urls.add(clean_url)
                is_video = ".mp4" in clean_url.lower() or "video" in clean_url.lower()
                yield (clean_url, is_video, f"Highlight Instagram (@{clean_user})")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
    except Exception as e:
        logger.debug(f"Không thể cào highlights Instagram cho @{clean_user}: {e}")


def resolve_facebook_profile_info(raw_input: str) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Trích xuất handle/username, tên hiển thị thật, user ID số và URL avatar chính chủ từ link Facebook hoặc username.
    Ví dụ: 'https://www.facebook.com/caca.tieu.50309/photos' -> ('caca.tieu.50309', 'Huỳnh Giang', '100050246662860', 'https://scontent...')
    """
    cleaned = raw_input.strip()
    m_id = re.search(r"profile\.php\?id=(\d+)", cleaned)
    if m_id:
        clean_handle = m_id.group(1)
    else:
        if "facebook.com/" in cleaned or "fb.com/" in cleaned:
            cleaned = re.sub(r"https?://(www\.|m\.|mbasic\.)?(facebook\.com|fb\.com)/", "", cleaned)
            cleaned = cleaned.split("/photos")[0].split("/posts")[0].split("?")[0].strip("/")
        clean_handle = cleaned.lstrip("@").strip()

    real_name = None
    user_id = None
    avatar_url = None

    profile_url = f"https://m.facebook.com/{clean_handle}"
    mobile_headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1"
        ),
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    try:
        req = urllib.request.Request(profile_url, headers=mobile_headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
            # og:title
            m_title = re.search(r"<meta\s+property=\"og:title\"\s+content=\"(.*?)\"", text)
            if m_title:
                t_val = html.unescape(m_title.group(1)).strip()
                if "Facebook" not in t_val:
                    real_name = t_val
            # user_id
            m_uid = re.search(r"fb://profile/(\d+)", text)
            if m_uid:
                user_id = m_uid.group(1)
            # avatar
            m_img = re.search(r"<meta\s+property=\"og:image\"\s+content=\"(.*?)\"", text)
            if m_img:
                avatar_url = html.unescape(m_img.group(1)).strip()
    except Exception as e:
        logger.debug(f"Không thể tra cứu thông tin Facebook profile @{clean_handle}: {e}")

    return clean_handle, real_name, user_id, avatar_url


def iter_facebook_images(query: str, max_count: Optional[int] = 40):
    """
    Generator tìm nạp từng URL ảnh thật công khai từ tài khoản hoặc link album ảnh Facebook.
    Hỗ trợ tải tất cả ảnh cho đến khi hết nếu max_count <= 0 hoặc None.
    Yields (url, is_avatar, caption) lần lượt ngay khi phát hiện để hiển thị trực tiếp lên UI.
    """
    clean_handle, real_name, user_id, avatar_url = resolve_facebook_profile_info(query)
    yielded = set()
    unlimited = (max_count is None or max_count <= 0)

    # 1. Phát avatar chính chủ làm mỏ neo đầu tiên nếu có
    if avatar_url and avatar_url not in yielded:
        yielded.add(avatar_url)
        yield (avatar_url, True, f"Avatar Facebook của {real_name or clean_handle}")

    # 2. Dùng gallery-dl để stream URL ảnh từ album Facebook
    q_strip = query.strip()
    if ("facebook.com" in q_strip or "fb.com" in q_strip) and q_strip.endswith("/photos"):
        target_fb_url = q_strip.replace("m.facebook.com", "www.facebook.com").replace("mbasic.facebook.com", "www.facebook.com")
    else:
        target_id = user_id or clean_handle
        target_fb_url = f"https://www.facebook.com/{target_id}/photos"

    try:
        cmd = [sys.executable, "-m", "gallery_dl", "--cookies-from-browser", "brave", "-g", target_fb_url]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line or not line.startswith("http"):
                    continue
                # Bỏ qua nếu là redirect link dẫn tới trang album khác
                if "facebook.com" in line and not any(ext in line.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    continue
                if line in yielded:
                    continue
                yielded.add(line)
                yield (line, False, f"Ảnh album Facebook ({real_name or clean_handle})")
                if not unlimited and len(yielded) >= max_count:
                    break
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
    except Exception as e:
        logger.debug(f"Lỗi khi cào ảnh Facebook qua gallery-dl: {e}")

    # 3. Fallback tìm kiếm web có định hướng nếu không cào được ảnh từ Facebook (do private hoặc bị block)
    if len(yielded) <= 1:
        fallback_cap = 60 if unlimited else max_count
        target_queries = []
        if real_name:
            target_queries.append(f'"{real_name}" chân dung')
            target_queries.append(f'"{real_name}" {clean_handle}')
        target_queries.append(f'"{clean_handle}" site:facebook.com')

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        for q_str in target_queries:
            if len(yielded) >= fallback_cap:
                break
            try:
                encoded_query = urllib.parse.quote(q_str)
                bing_url = f"https://www.bing.com/images/async?q={encoded_query}&count=35"
                req = urllib.request.Request(bing_url, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as res:
                    html_text = res.read().decode("utf-8", errors="ignore")
                    bing_imgs = re.findall(r'murl&quot;:&quot;(https?://[^&"\s]+)&quot;', html_text)
                    for b_u in bing_imgs:
                        if b_u not in yielded and b_u.startswith("http"):
                            yielded.add(b_u)
                            yield (b_u, False, f"Web: {q_str}")
                            if len(yielded) >= fallback_cap:
                                break
            except Exception as e:
                logger.debug(f"Lỗi tìm kiếm Bing cho query '{q_str}': {e}")


def search_facebook_images(query: str, count: int = 30) -> List[str]:
    """Tìm nạp danh sách URL ảnh công khai từ tài khoản hoặc link Facebook."""
    max_c = 0 if count <= 0 else count * 2
    return [item[0] for item in iter_facebook_images(query, max_count=max_c)]


def download_and_verify_image(
    url: str,
    min_size: int = 120,
    timeout: int = 5
) -> Optional[Image.Image]:
    """Tải và xác thực tệp ảnh từ URL, kiểm tra kích thước tối thiểu và timeout nhanh."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            data = res.read()
            if len(data) < 5000:  # Quá nhỏ (< 5KB), thường là icon/tracker
                return None
            img = Image.open(BytesIO(data))
            img.load()
            w, h = img.size
            if w < min_size or h < min_size:
                return None
            return img.convert("RGB")
    except Exception:
        return None


def preload_anchor_from_dir(dir_path: Path, engine_obj, max_samples: int = 5) -> Optional[np.ndarray]:
    """Nạp vector mỏ neo từ các ảnh có sẵn trong thư mục nếu có."""
    if not dir_path.exists() or not dir_path.is_dir():
        return None
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    existing_imgs = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts][:max_samples]
    if not existing_imgs or engine_obj is None:
        return None

    embs = []
    for f in existing_imgs:
        e = engine_obj.extract_from_file(f, filter_quality=True, select_single=True)
        if e:
            embs.append(e[0])

    if embs:
        mat = np.array(embs)
        c = np.mean(mat, axis=0)
        norm = np.linalg.norm(c)
        if norm > 0:
            return c / norm
    return None


def download_faces(
    query: str,
    output_dir: Path,
    target_count: int = 10,
    prefix: Optional[str] = None,
    verify_face: bool = True,
    same_person: bool = True,
    similarity_threshold: float = 0.38,
    anchor_seed_count: int = 5,
    engine_obj=None,
    is_instagram: bool = False,
    is_facebook: bool = False,
) -> int:
    """
    Tìm kiếm và tải về ảnh đạt chuẩn khuôn mặt, tự động phân trang liên tục cho đến khi đủ số lượng.
    Đồng thời tự động trích xuất vector khuôn mặt mỏ neo để đảm bảo TẤT CẢ các ảnh sau đó đều là CÙNG 1 NGƯỜI.
    Hỗ trợ chế độ tải trực tiếp từ Instagram (@username) hoặc Facebook profile/album link.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    is_fb_mode = is_facebook or ("facebook.com" in query) or ("fb.com" in query)
    is_ig_mode = not is_fb_mode and (is_instagram or query.startswith("@") or "instagram.com" in query)
    is_unlimited = (target_count <= 0)

    if is_fb_mode:
        clean_handle, fb_name, _uid, _av = resolve_facebook_profile_info(query)
        target_str = "Toàn bộ album (cho đến khi hết ảnh)" if is_unlimited else f"{target_count} ảnh"
        click.echo(f"📘 Bắt đầu cào ảnh Facebook: '{fb_name or clean_handle}' (@{clean_handle}) (Mục tiêu: {target_str})...")
    elif is_ig_mode:
        clean_user = query.strip().lstrip("@").split("instagram.com/")[-1].strip("/").split("?")[0]
        target_str = "Toàn bộ profile (cho đến khi hết ảnh)" if is_unlimited else f"{target_count} ảnh"
        click.echo(f"📸 Bắt đầu cào ảnh Instagram từ tài khoản: '@{clean_user}' (Mục tiêu: {target_str})...")
    else:
        target_str = "Tối đa có thể" if is_unlimited else f"{target_count} ảnh"
        click.echo(f"🔍 Bắt đầu tìm kiếm ảnh cho: '{query}' (Mục tiêu: {target_str})...")

    if same_person:
        click.echo(f"🎯 Chế độ đồng nhất danh tính: BẬT (Ngưỡng ArcFace sim >= {similarity_threshold})")

    clean_prefix = prefix or re.sub(r"[^a-zA-Z0-9_]+", "_", query.lower()).strip("_")

    # Xác định chỉ số tiếp theo trên đĩa để không ghi đè ảnh cũ
    existing_files = [f.stem for f in output_dir.iterdir() if f.is_file()]
    idx = 1
    while f"{clean_prefix}_{idx:04d}" in existing_files:
        idx += 1

    saved_count = 0
    seen_urls: Set[str] = set()
    seed_embeddings: List[np.ndarray] = []

    # 1. Thử nạp mỏ neo từ các ảnh đã có sẵn trong thư mục
    anchor_centroid = None
    if same_person and engine_obj is not None:
        anchor_centroid = preload_anchor_from_dir(output_dir, engine_obj)
        if anchor_centroid is not None:
            click.echo(f"⚡ Đã nạp vector khuôn mặt mỏ neo từ ảnh có sẵn trong '{output_dir.name}/'!")

    # 2. Xây dựng danh sách nguồn URL ảnh mạng xã hội nếu có
    social_urls: List[str] = []
    if is_fb_mode:
        social_urls = search_facebook_images(query, count=0 if is_unlimited else target_count * 3)
        click.echo(f"  ✨ Đã tìm nạp được {len(social_urls)} ảnh từ Facebook profile/album.")
    elif is_ig_mode:
        social_urls = search_instagram_images(query, count=0 if is_unlimited else target_count * 3)
        click.echo(f"  ✨ Đã tìm nạp được {len(social_urls)} ảnh từ Instagram profile.")

    # Các biến thể tìm kiếm web mở rộng
    search_queries = [
        query,
        f"{query} portrait",
        f"{query} face photo",
        f"{query} interview",
    ]

    with tqdm(total=None if is_unlimited else target_count, desc="Tiến độ thu thập", unit="ảnh") as pbar:
        # Xử lý các link Facebook/Instagram đã tìm nạp trước
        if social_urls:
            source_label = "Facebook" if is_fb_mode else "Instagram"
            pbar.set_postfix_str(f"Đang kiểm tra ảnh từ {source_label}...")
            for u_idx, url in enumerate(social_urls, start=1):
                if not is_unlimited and saved_count >= target_count:
                    break
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                img = download_and_verify_image(url, timeout=5)
                if img is None:
                    continue

                face_enc = None
                if verify_face and engine_obj is not None:
                    arr = np.array(img)
                    embs = engine_obj.extract_face_embeddings(arr, filter_quality=True, select_single=True)
                    if not embs:
                        continue
                    face_enc = embs[0]

                if same_person and face_enc is not None:
                    if anchor_centroid is None:
                        seed_embeddings.append(face_enc)
                        if len(seed_embeddings) >= min(3, anchor_seed_count):
                            s_mat = np.array(seed_embeddings)
                            sim_mat = s_mat @ s_mat.T
                            avg_sims = np.mean(sim_mat, axis=1)
                            c = np.mean(s_mat[avg_sims >= np.median(avg_sims)], axis=0)
                            anchor_centroid = c / np.linalg.norm(c)
                            click.echo(f"\n🎯 [Mỏ neo {source_label}] Đã xác lập vector nhận diện chuẩn cho '{query}'!")
                    else:
                        sim = float(np.dot(anchor_centroid, face_enc))
                        if sim < similarity_threshold:
                            continue
                        anchor_centroid = 0.95 * anchor_centroid + 0.05 * face_enc
                        anchor_centroid = anchor_centroid / np.linalg.norm(anchor_centroid)

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
                    pbar.update(1)
                    target_display = "toàn bộ" if is_unlimited else str(target_count)
                    if same_person and anchor_centroid is not None and face_enc is not None:
                        match_sim = float(np.dot(anchor_centroid, face_enc))
                        click.echo(f"  📥 [{saved_count}/{target_display}] Đúng người (sim={match_sim * 100:.1f}%) ✅ -> Đã lưu: {filename}")
                    else:
                        click.echo(f"  📥 [{saved_count}/{target_display}] Đạt chuẩn mặt ✅ -> Đã lưu: {filename}")
                except Exception as e:
                    logger.debug(f"Lỗi khi lưu ảnh {save_path}: {e}")

        # Nếu là chế độ mạng xã hội (Facebook/Instagram) và đã cào xong, không fallback sang web
        if (is_fb_mode or is_ig_mode) and (is_unlimited or saved_count > 0):
            return saved_count

        # Nếu chưa đủ target_count, tiếp tục tìm kiếm trên Bing Search
        for q in search_queries:
            if not is_unlimited and saved_count >= target_count:
                break

            first = 0
            empty_pages_in_a_row = 0

            # Lật trang tìm kiếm liên tục (mỗi trang ~35 ảnh)
            while first <= 400 and (is_unlimited or saved_count < target_count) and empty_pages_in_a_row < 2:
                pbar.set_postfix_str(f"Đang tìm trang mới (vị trí: {first})")
                new_urls = search_image_urls_page(q, first=first, count=35)

                if not new_urls:
                    empty_pages_in_a_row += 1
                    first += 35
                    continue

                empty_pages_in_a_row = 0
                urls_to_process = [u for u in new_urls if u not in seen_urls]
                for u in urls_to_process:
                    seen_urls.add(u)

                first += 35

                # Tải và kiểm duyệt từng link trong trang vừa lấy
                for u_idx, url in enumerate(urls_to_process, start=1):
                    if saved_count >= target_count:
                        break

                    pbar.set_postfix_str(f"Đang kiểm tra link {u_idx}/{len(urls_to_process)}")
                    img = download_and_verify_image(url, timeout=4)
                    if img is None:
                        continue

                    # Kiểm duyệt chất lượng khuôn mặt bằng InsightFace
                    face_enc = None
                    if verify_face and engine_obj is not None:
                        arr = np.array(img)
                        embs = engine_obj.extract_face_embeddings(arr, filter_quality=True, select_single=True)
                        if not embs:
                            pbar.set_postfix_str("Bỏ qua (Không phát hiện mặt rõ)")
                            continue
                        face_enc = embs[0]

                    # Xác thực đồng nhất danh tính (Cùng 1 người)
                    if same_person and face_enc is not None:
                        if anchor_centroid is None:
                            # Thu thập 5 ảnh đầu làm seed
                            seed_embeddings.append(face_enc)
                            if len(seed_embeddings) >= anchor_seed_count:
                                seed_mat = np.array(seed_embeddings)
                                sim_mat = seed_mat @ seed_mat.T
                                avg_sims = np.mean(sim_mat, axis=1)
                                core_mask = avg_sims >= np.median(avg_sims)
                                core_seeds = seed_mat[core_mask]
                                c = np.mean(core_seeds, axis=0)
                                anchor_centroid = c / np.linalg.norm(c)
                                core_sim = float(np.mean(avg_sims[core_mask]))
                                click.echo(f"\n🎯 [Mỏ neo danh tính] Đã xác lập vector nhận diện chuẩn cho '{query}' (độ tương đồng nòng cốt: {core_sim * 100:.1f}%)!")
                        else:
                            sim = float(np.dot(anchor_centroid, face_enc))
                            if sim < similarity_threshold:
                                pbar.set_postfix_str(f"Bỏ qua (Khác người, sim={sim*100:.1f}%)")
                                continue

                            # Cập nhật mỏ neo nhẹ (momentum 95/5) thích ứng các góc mặt
                            anchor_centroid = 0.95 * anchor_centroid + 0.05 * face_enc
                            anchor_centroid = anchor_centroid / np.linalg.norm(anchor_centroid)

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
                        pbar.update(1)
                        target_display = "toàn bộ" if is_unlimited else str(target_count)
                        if same_person and anchor_centroid is not None and face_enc is not None:
                            match_sim = float(np.dot(anchor_centroid, face_enc))
                            click.echo(f"  📥 [{saved_count}/{target_display}] Đúng người (sim={match_sim * 100:.1f}%) ✅ -> Đã lưu: {filename}")
                        else:
                            click.echo(f"  📥 [{saved_count}/{target_display}] Đạt chuẩn mặt ✅ -> Đã lưu: {filename}")
                    except Exception as e:
                        logger.debug(f"Lỗi khi lưu ảnh {save_path}: {e}")

    return saved_count
