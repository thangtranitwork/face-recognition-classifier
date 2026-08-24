#!/usr/bin/env bash
# ============================================================
# Face Recognition File Renamer - Runner Script
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"
MAIN="$SCRIPT_DIR/rename_faces.py"
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
    echo "  ║   Face Recognition File Renamer          ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_usage() {
    echo -e "${BOLD}Cách dùng:${NC}"
    echo "  ./run.sh [OPTION]"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo "  (không có)     Dry-run — xem kết quả, KHÔNG đổi tên"
    echo "  --apply        Thực sự đổi tên file"
    echo "  --clear-cache  Xóa cache và build lại"
    echo "  --setup        Cài đặt dependencies (chỉ cần chạy lần đầu)"
    echo "  --help         Hiện thông báo này"
    echo ""
    echo -e "${BOLD}Ví dụ:${NC}"
    echo "  ./run.sh               # preview kết quả"
    echo "  ./run.sh --apply       # đổi tên thật"
    echo "  ./run.sh --clear-cache # build lại cache"
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
        info "Đang tiến hành đổi tên file thực sự..."
        "$PYTHON" "$MAIN" --apply
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
        ;;

    *)
        error "Tùy chọn không hợp lệ: $1"
        echo ""
        print_usage
        exit 1
        ;;
esac
