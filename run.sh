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
CLUSTER="$SCRIPT_DIR/cluster_faces.py"
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
    echo -e "${BOLD}Options cho GOM NHÓM NGƯỜI LẠ (Auto-Clustering):${NC}"
    echo "  --cluster         Dry-run — phân tích & xem các nhóm người lạ xuất hiện nhiều lần"
    echo "  --cluster-apply   Tự động TẠO FOLDER NGƯỜI MỚI (nguoi_moi_1, 2...) và di chuyển file vào"
    echo ""
    echo -e "${BOLD}Options cho KIỂM THỬ (Test):${NC}"
    echo "  --prepare-test    Reset dataset & lấy ngẫu nhiên N ảnh sang chua_phan_loai để test"
    echo ""
    echo -e "${BOLD}Hệ thống:${NC}"
    echo "  --setup           Cài đặt dependencies (chạy lần đầu)"
    echo "  --help            Hiện thông báo này"
    echo ""
    echo -e "${BOLD}Ví dụ:${NC}"
    echo "  ./run.sh                  # Preview kết quả đổi tên"
    echo "  ./run.sh --apply          # Đổi tên file thật"
    echo "  ./run.sh --cluster-apply  # Gom nhóm người lạ & tự tạo folder mới"
    echo "  ./run.sh --organize-apply # Chuyển file đã đổi tên vào folder từng người (trừ unknown)"
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

# ── Argument Parser ─────────────────────────────────────────
TARGET_DIR=""
ACTION="dry-run"
PY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            print_usage
            exit 0
            ;;
        --setup)
            do_setup
            exit 0
            ;;
        --apply)
            ACTION="apply"
            shift
            ;;
        --organize|--move)
            ACTION="organize"
            shift
            ;;
        --organize-apply|--move-apply)
            ACTION="organize-apply"
            shift
            ;;
        --cluster)
            ACTION="cluster"
            shift
            ;;
        --cluster-apply)
            ACTION="cluster-apply"
            shift
            ;;
        --prepare-test)
            ACTION="prepare-test"
            shift
            ;;
        --clear-cache)
            ACTION="clear-cache"
            shift
            ;;
        -d|--dir|--target-dir)
            TARGET_DIR="$2"
            PY_ARGS+=("-d" "$2")
            shift 2
            ;;
        *)
            # Nếu người dùng gõ trực tiếp tên folder, vd: ./run.sh other hoặc ./run.sh --cluster other
            if [ -z "$TARGET_DIR" ] && [ -d "$SCRIPT_DIR/dataset/$1" -o -d "$1" ]; then
                TARGET_DIR="$1"
                PY_ARGS+=("-d" "$1")
                shift
            else
                error "Tùy chọn không hợp lệ: $1"
                echo ""
                print_usage
                exit 1
            fi
            ;;
    esac
done

check_env

case "$ACTION" in
    apply)
        info "Đang tiến hành ĐỔI TÊN file thực sự..."
        "$PYTHON" "$MAIN" --apply "${PY_ARGS[@]}"
        ;;

    organize)
        info "Dry-run di chuyển file đã đổi tên..."
        "$PYTHON" "$ORGANIZE" "${PY_ARGS[@]}"
        ;;

    organize-apply)
        info "Đang DI CHUYỂN các file đã đổi tên vào folder tương ứng (trừ unknown_*)..."
        "$PYTHON" "$ORGANIZE" --apply "${PY_ARGS[@]}"
        ;;

    cluster)
        info "Dry-run gom nhóm người lạ..."
        "$PYTHON" "$CLUSTER" "${PY_ARGS[@]}"
        ;;

    cluster-apply)
        info "Đang GOM NHÓM & TẠO FOLDER NGƯỜI MỚI tự động..."
        "$PYTHON" "$CLUSTER" --apply "${PY_ARGS[@]}"
        ;;

    prepare-test)
        info "Đang chuyển ảnh mẫu sang thư mục chua_phan_loai để test..."
        "$PYTHON" "$PREPARE" "${PY_ARGS[@]}"
        ;;

    clear-cache)
        info "Xóa cache và chạy lại..."
        "$PYTHON" "$MAIN" --clear-cache "${PY_ARGS[@]}"
        ;;

    dry-run)
        info "Chế độ DRY-RUN (chỉ xem kết quả, không đổi tên)"
        echo ""
        "$PYTHON" "$MAIN" "${PY_ARGS[@]}"
        echo ""
        echo -e "${YELLOW}💡 Nếu hài lòng, chạy: ./run.sh --apply ${PY_ARGS[*]:-}${NC}"
        echo -e "${YELLOW}💡 Để gom nhóm người lạ: ./run.sh --cluster-apply ${PY_ARGS[*]:-}${NC}"
        echo -e "${YELLOW}💡 Để chuyển file về đúng folder người: ./run.sh --organize-apply ${PY_ARGS[*]:-}${NC}"
        ;;
esac
