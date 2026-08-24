#!/usr/bin/env bash
# ============================================================
# Face Recognition File Renamer - Runner Script
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"
MAIN="$SCRIPT_DIR/rename_faces.py"
ORGANIZE="$SCRIPT_DIR/organize_faces.py"
PREPARE="$SCRIPT_DIR/prepare_test_data.py"
CONFIG="$SCRIPT_DIR/config.yaml"

# ── Màu sắc ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Hàm tiện ích ────────────────────────────────────────────
info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*" >&2; }
die()     { error "$*"; exit 1; }

print_banner() {
    echo -e "${BOLD}"
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║   Face Recognition Classifier Tool       ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_usage() {
    echo -e "${BOLD}Cách dùng:${NC}"
    echo "  ./run.sh [OPTION]"
    echo ""
    echo -e "${BOLD}Options cho ĐỔI TÊN (Rename):${NC}"
    echo "  (không có)        Dry-run — xem trước kết quả đổi tên"
    echo "  --apply           Thực sự ĐỔI TÊN các file trong chua_phan_loai"
    echo "  --clear-cache     Xóa cache encodings và build lại"
    echo ""
    echo -e "${BOLD}Options cho DI CHUYỂN (Organize / Move):${NC}"
    echo "  --organize        Dry-run — xem trước việc di chuyển file đã đổi tên"
    echo "  --organize-apply  Thực sự DI CHUYỂN file đã đổi tên vào đúng folder người (TRỪ unknown_*)"
    echo ""
    echo -e "${BOLD}Options cho KIỂM THỬ (Test):${NC}"
    echo "  --prepare-test    Lấy ngẫu nhiên N ảnh từ dataset chuyển sang chua_phan_loai để test"
    echo ""
    echo -e "${BOLD}Hệ thống:${NC}"
    echo "  --setup           Cài đặt dependencies (chạy lần đầu)"
    echo "  --help            Hiện thông báo này"
    echo ""
    echo -e "${BOLD}Ví dụ:${NC}"
    echo "  ./run.sh                  # Preview kết quả đổi tên"
    echo "  ./run.sh --apply          # Đổi tên file thật"
    echo "  ./run.sh --organize-apply # Chuyển file đã đổi tên vào folder từng người (trừ unknown)"
    echo "  ./run.sh --prepare-test   # Lấy ngẫu nhiên ảnh để test"
}

# ── Setup: tạo venv + cài packages ──────────────────────────
do_setup() {
    info "Đang setup môi trường..."

    if [ ! -d "$VENV" ]; then
        info "Tạo virtual environment..."
        python3 -m venv "$VENV"
        success "Tạo .venv xong"
    else
        info ".venv đã tồn tại, bỏ qua"
    fi

    info "Cài dependencies..."
    "$VENV/bin/pip" install --upgrade pip -q
    "$VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

    # Patch face_recognition_models nếu cần (Python 3.14 compatibility)
    MODELS_INIT="$VENV/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/face_recognition_models/__init__.py"
    if [ -f "$MODELS_INIT" ] && grep -q "pkg_resources" "$MODELS_INIT"; then
        warn "Phát hiện Python 3.14+: patch face_recognition_models..."
        python3 - "$MODELS_INIT" << 'PYEOF'
import sys
path = sys.argv[1]
new_content = '''# -*- coding: utf-8 -*-
import os as _os
_models_dir = _os.path.join(_os.path.dirname(__file__), "models")

def pose_predictor_model_location():
    return _os.path.join(_models_dir, "shape_predictor_68_face_landmarks.dat")

def pose_predictor_five_point_model_location():
    return _os.path.join(_models_dir, "shape_predictor_5_face_landmarks.dat")

def face_recognition_model_location():
    return _os.path.join(_models_dir, "dlib_face_recognition_resnet_model_v1.dat")

def cnn_face_detector_model_location():
    return _os.path.join(_models_dir, "mmod_human_face_detector.dat")
'''
with open(path, 'w') as f:
    f.write(new_content)
print("  ✅ Patch xong")
PYEOF
    fi

    success "Setup hoàn tất! Chạy './run.sh' để bắt đầu."
}

# ── Kiểm tra môi trường ──────────────────────────────────────
check_env() {
    if [ ! -f "$PYTHON" ]; then
        die "Chưa có virtual environment. Chạy: ./run.sh --setup"
    fi
    if [ ! -f "$MAIN" ]; then
        die "Không tìm thấy $MAIN"
    fi
    if [ ! -f "$CONFIG" ]; then
        die "Không tìm thấy $CONFIG"
    fi
}

# ── Main ─────────────────────────────────────────────────────
print_banner

case "${1:-}" in
    --help|-h)
        print_usage
        ;;

    --setup)
        do_setup
        ;;

    --apply)
        check_env
        info "Đang tiến hành ĐỔI TÊN file thực sự..."
        "$PYTHON" "$MAIN" --apply
        ;;

    --organize|--move)
        check_env
        info "Dry-run di chuyển file đã đổi tên..."
        "$PYTHON" "$ORGANIZE"
        ;;

    --organize-apply|--move-apply)
        check_env
        info "Đang DI CHUYỂN các file đã đổi tên vào folder tương ứng (trừ unknown_*)..."
        "$PYTHON" "$ORGANIZE" --apply
        ;;

    --prepare-test)
        check_env
        info "Đang chuyển ảnh mẫu sang thư mục chua_phan_loai để test..."
        "$PYTHON" "$PREPARE"
        ;;

    --clear-cache)
        check_env
        info "Xóa cache và chạy lại..."
        "$PYTHON" "$MAIN" --clear-cache
        ;;

    "")
        check_env
        info "Chế độ DRY-RUN (chỉ xem kết quả, không đổi tên)"
        echo ""
        "$PYTHON" "$MAIN"
        echo ""
        echo -e "${YELLOW}💡 Nếu hài lòng, chạy: ./run.sh --apply${NC}"
        echo -e "${YELLOW}💡 Để chuyển file đã đổi tên về đúng folder người: ./run.sh --organize-apply${NC}"
        ;;

    *)
        error "Tùy chọn không hợp lệ: $1"
        echo ""
        print_usage
        exit 1
        ;;
esac
