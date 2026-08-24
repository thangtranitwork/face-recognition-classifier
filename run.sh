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
    echo "  ./run.sh [FLAGS]"
    echo ""
    echo -e "${BOLD}Flags tính năng (có thể viết tắt & ghép chung):${NC}"
    echo "  -r, --rename            Nhận diện & đổi tên ảnh (Mặc định)"
    echo "  -c, --cluster           Gom nhóm tự động khuôn mặt người lạ (DBSCAN)"
    echo "  -o, -m, --organize      Di chuyển các file đã đổi tên về đúng thư mục người"
    echo "  -p, --prepare-test      Reset dataset & lấy ảnh mẫu sang chua_phan_loai để test"
    echo "  -cc, --clear-cache      Xóa cache encodings và build lại từ đầu"
    echo ""
    echo -e "${BOLD}Flag thực thi:${NC}"
    echo "  -a, --apply             Thực sự ĐỔI TÊN / TẠO FOLDER / MOVE (Mặc định là Dry-Run)"
    echo ""
    echo -e "${BOLD}Tham số thư mục:${NC}"
    echo "  -d, --dir, <tên>        Chỉ định thư mục cần chạy (vd: -d other, -d chua_phan_loai)"
    echo ""
    echo -e "${BOLD}Ví dụ cú pháp viết tắt siêu ngắn:${NC}"
    echo "  ./run.sh -a                       # Đổi tên thật (-a)"
    echo "  ./run.sh -c                       # Dry-run gom nhóm người lạ (-c)"
    echo "  ./run.sh -ca                      # Gom nhóm người lạ & tạo folder mới (-c -a)"
    echo "  ./run.sh -cao                     # Gom nhóm người mới + Chuyển file về folder người (-c -a -o)"
    echo "  ./run.sh -cao -d other            # Chạy gom nhóm & di chuyển trên folder 'other'"
    echo "  ./run.sh -ao                      # Đổi tên thật & chuyển file về folder người (-a -o)"
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

DO_RENAME=false
DO_CLUSTER=false
DO_ORGANIZE=false
DO_PREPARE=false
DO_CLEAR_CACHE=false
IS_APPLY=false
IS_YES=false
TARGET_DIR=""
DIR_ARGS=()

MODULE_COUNT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir|--target-dir|-dir)
            TARGET_DIR="$2"
            DIR_ARGS=("-d" "$2")
            shift 2
            continue
            ;;
    esac

    # Tự động tách các short flags ghép chung (ví dụ: -cao -> -c -a -o)
    if [[ "$1" =~ ^-[a-z]{2,}$ ]] && [[ "$1" != "-cc" ]]; then
        flags="${1#-}"
        shift
        expanded=()
        for (( i=0; i<${#flags}; i++ )); do
            expanded+=("-${flags:$i:1}")
        done
        set -- "${expanded[@]}" "$@"
        continue
    fi

    case "$1" in
        --help|-h)
            print_usage
            exit 0
            ;;
        --setup)
            do_setup
            exit 0
            ;;
        --apply|-a)
            IS_APPLY=true
            shift
            ;;
        --yes|-y)
            IS_YES=true
            shift
            ;;
        --rename|-r)
            DO_RENAME=true
            MODULE_COUNT=$((MODULE_COUNT + 1))
            shift
            ;;
        --cluster|-c|--cluster-apply)
            DO_CLUSTER=true
            if [ "$1" = "--cluster-apply" ]; then
                IS_APPLY=true
            fi
            MODULE_COUNT=$((MODULE_COUNT + 1))
            shift
            ;;
        --organize|-o|--move|-m|--organize-apply|--move-apply)
            DO_ORGANIZE=true
            if [ "$1" = "--organize-apply" -o "$1" = "--move-apply" ]; then
                IS_APPLY=true
            fi
            MODULE_COUNT=$((MODULE_COUNT + 1))
            shift
            ;;
        --prepare-test|-p)
            DO_PREPARE=true
            MODULE_COUNT=$((MODULE_COUNT + 1))
            shift
            ;;
        --clear-cache|-cc)
            DO_CLEAR_CACHE=true
            shift
            ;;
        *)
            # Nếu truyền trực tiếp tên folder (vd: ./run.sh other)
            if [ -z "$TARGET_DIR" ] && [ -d "$SCRIPT_DIR/dataset/$1" -o -d "$1" ]; then
                TARGET_DIR="$1"
                DIR_ARGS=("-d" "$1")
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

# Nếu người dùng không chọn module tính năng nào -> Mặc định là --rename
if [ "$MODULE_COUNT" -eq 0 ] && [ "$DO_CLEAR_CACHE" = false ]; then
    DO_RENAME=true
fi

check_env

# ── 1. Clear cache nếu được gọi ─────────────────────────────
if [ "$DO_CLEAR_CACHE" = true ]; then
    info "🗑️  Xóa cache encodings..."
    "$PYTHON" "$MAIN" --clear-cache "${DIR_ARGS[@]}"
fi

# ── 2. Reset / Prepare test data ────────────────────────────
if [ "$DO_PREPARE" = true ]; then
    info "🧪 Chuẩn bị dữ liệu kiểm thử..."
    "$PYTHON" "$PREPARE" "${DIR_ARGS[@]}"
fi

APPLY_FLAG=()
if [ "$IS_APPLY" = true ]; then
    APPLY_FLAG=("--apply")
fi

YES_FLAG=()
if [ "$IS_YES" = true ]; then
    YES_FLAG=("-y")
fi

# ── 3. Rename faces ─────────────────────────────────────────
if [ "$DO_RENAME" = true ]; then
    if [ "$IS_APPLY" = true ]; then
        info "✏️  Đang tiến hành ĐỔI TÊN file..."
    else
        info "🔍 Dry-run ĐỔI TÊN file..."
    fi
    "$PYTHON" "$MAIN" "${APPLY_FLAG[@]}" "${DIR_ARGS[@]}"
    echo ""
fi

# ── 4. Auto-cluster unknown faces ───────────────────────────
if [ "$DO_CLUSTER" = true ]; then
    if [ "$IS_APPLY" = true ]; then
        info "🤖 Đang GOM NHÓM người mới tự động..."
    else
        info "🤖 Dry-run GOM NHÓM người lạ..."
    fi
    "$PYTHON" "$CLUSTER" "${APPLY_FLAG[@]}" "${DIR_ARGS[@]}"
    echo ""
fi

# ── 5. Organize / Move files to person folders ──────────────
if [ "$DO_ORGANIZE" = true ]; then
    if [ "$IS_APPLY" = true ]; then
        info "🚚 Đang DI CHUYỂN các file về đúng folder người (trừ unknown_*)..."
    else
        info "🔍 Dry-run DI CHUYỂN file về folder người..."
    fi
    "$PYTHON" "$ORGANIZE" "${APPLY_FLAG[@]}" "${YES_FLAG[@]}" "${DIR_ARGS[@]}"
    echo ""
fi

if [ "$IS_APPLY" = false ]; then
    echo -e "${YELLOW}💡 Chế độ DRY-RUN (xem thử). Để thực sự đổi tên/tạo folder/move file, thêm flag --apply${NC}"
fi
