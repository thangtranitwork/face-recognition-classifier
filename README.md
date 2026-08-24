# Face Recognition File Renamer

Tool tự động nhận diện khuôn mặt và đổi tên ảnh trong thư mục chưa phân loại.

## Cấu trúc thư mục

```
face-recognition-classifier/
├── rename_faces.py       # Script chính
├── config.yaml           # Cấu hình
├── requirements.txt      # Dependencies
└── dataset/              # Đặt dataset của bạn ở đây
    ├── nguoi_A/          # Ảnh đã phân loại của người A (≥9 ảnh)
    ├── nguoi_B/
    ├── nguoi_C/
    ├── khac/             # Folder bị exclude (trong config)
    └── chua_phan_loai/   # Ảnh sẽ được đổi tên
```

## Cài đặt

```bash
# 1. Tạo virtualenv (khuyến nghị)
python3 -m venv .venv
source .venv/bin/activate

# 2. Cài dlib (dependency của face_recognition)
# Ubuntu/Debian:
sudo apt-get install -y cmake libopenblas-dev liblapack-dev

# 3. Cài packages
pip install -r requirements.txt
```

## Sử dụng

```bash
# Dry-run: xem kết quả nhận diện mà không đổi tên
python rename_faces.py

# Thực sự đổi tên
python rename_faces.py --apply

# Xóa cache và build lại (khi thêm ảnh training mới)
python rename_faces.py --clear-cache

# Chỉ định thư mục chưa phân loại khác (ghi đè config)
python rename_faces.py --unclassified-dir "folder_khac" --apply

# Dùng file config khác
python rename_faces.py --config /path/to/other_config.yaml
```

## Kết quả đổi tên

| Trường hợp | Tên file gốc | Tên file mới |
|---|---|---|
| Nhận diện được | `IMG_001.jpg` | `nguoi_A_IMG_001.jpg` |
| Không nhận diện | `IMG_002.jpg` | `unknown_IMG_002.jpg` |
| Nhiều khuôn mặt | `IMG_003.jpg` | `unknown_IMG_003.jpg` |
| Không có mặt | `IMG_004.jpg` | `unknown_IMG_004.jpg` |

## Cấu hình quan trọng (`config.yaml`)

| Setting | Mặc định | Mô tả |
|---|---|---|
| `exclude_dirs` | `[chua_phan_loai, khac]` | Folder bỏ qua khi load known faces |
| `tolerance` | `0.55` | Ngưỡng nhận diện (thấp hơn = nghiêm hơn) |
| `max_images_per_person` | `50` | Giới hạn ảnh training/người (null = lấy hết) |
| `model` | `hog` | `hog` (nhanh/CPU) hoặc `cnn` (chính xác/GPU) |
| `use_cache` | `true` | Cache encodings để chạy nhanh lần sau |

## Lưu ý

- **Cache**: Lần đầu chạy sẽ chậm (encode tất cả ảnh training). Lần sau dùng cache nhanh hơn nhiều. Cache tự invalidate khi dataset thay đổi.
- **Tolerance**: Nếu nhận diện sai nhiều → giảm tolerance (0.5). Nếu bỏ sót nhiều → tăng lên (0.6).
- **Ảnh training**: Mỗi người nên có ảnh đa dạng góc chụp, ánh sáng để nhận diện chính xác hơn.
- **Dry-run luôn trước**: Xem log để kiểm tra kết quả trước khi chạy `--apply`.
