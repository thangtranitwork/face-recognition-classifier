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
NORMALIZE="$SCRIPT_DIR/clean_names.py"
DOWNLOAD="$SCRIPT_DIR/download_faces.py"
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
    echo "  -n, --clean, --norm     Chuẩn hóa tên file trong thư mục người (thành <tên>_<STT>)"
    echo "  -p, --prepare-test      Reset dataset & lấy ảnh mẫu sang chua_phan_loai để test"
    echo "  -u, --ui, --studio      Khởi chạy giao diện Web Studio (Dashboard, trực quan hóa)"
    echo "  -dl, --download <tên>   Tải thêm ảnh từ web cho 1 người (hoặc người lạ)
  --ig, --instagram       Cào ảnh trực tiếp từ tài khoản Instagram công khai (@username)"
    echo "  -cc, --clear-cache      Xóa cache encodings và build lại từ đầu"
    echo ""
    echo -e "${BOLD}Flag tham số bổ trợ:${NC}"
    echo "  -a, --apply             Thực sự ĐỔI TÊN / TẠO FOLDER / MOVE (Mặc định là Dry-Run)"
    echo "  -y, --yes               Tự động chấp nhận tên mặc định khi tạo folder người mới"
    echo "  --count, -count <số>    Số lượng ảnh lấy mẫu khi prepare (-p) hoặc tải về (-dl)"
    echo "  --port <cổng>           Chỉ định cổng cho Web Studio (Mặc định: 8000)"
    echo "  -d, --dir, <tên>        Chỉ định thư mục cần chạy (vd: -d other, -d binhan)"
    echo ""
    echo -e "${BOLD}Ví dụ cú pháp thực tế:${NC}"
    echo "  ./run.sh --ui                     # Khởi chạy giao diện Web Studio tại http://localhost:8000"
    echo "  ./run.sh -p --count 5             # Tạo ~75 ảnh test ẩn danh sạch sẽ từ dataset"
    echo "  ./run.sh -a                       # Đổi tên thật trong chua_phan_loai"
    echo "  ./run.sh -c                       # Dry-run gom nhóm người lạ (-c)"
    echo "  ./run.sh -cao                     # Gom nhóm người mới + Chuyển file về folder người"
    echo "  ./run.sh -dl 'Taylor Swift' -p chua_phan_loai --count 5  # Tải 5 ảnh người lạ để test gom nhóm"
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
DO_NORMALIZE=false
DO_PREPARE=false
DO_CLEAR_CACHE=false
DO_DOWNLOAD=false
DO_UI=false
DO_IG=false
DO_FB=false
UI_PORT=8000
UI_HOST="0.0.0.0"
DOWNLOAD_QUERY=""
SAMPLE_COUNT=""
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
        --count|-count)
            SAMPLE_COUNT="$2"
            shift 2
            continue
            ;;
        --port)
            UI_PORT="$2"
            shift 2
            continue
            ;;
        --download|-dl)
            DO_DOWNLOAD=true
            DOWNLOAD_QUERY="$2"
            MODULE_COUNT=$((MODULE_COUNT + 1))
            shift 2
            continue
            ;;
    esac

    # Tự động tách các short flags ghép chung (ví dụ: -cao -> -c -a -o)
    if [[ "$1" =~ ^-[a-z]{2,}$ ]] && [[ "$1" != "-cc" ]] && [[ "$1" != "-dl" ]]; then
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
        --clean|-n|--normalize|--norm)
            DO_NORMALIZE=true
            MODULE_COUNT=$((MODULE_COUNT + 1))
            shift
            ;;
        --prepare-test|-p)
            DO_PREPARE=true
            MODULE_COUNT=$((MODULE_COUNT + 1))
            shift
            ;;
        --ui|-u|--studio|-s)
            DO_UI=true
            MODULE_COUNT=$((MODULE_COUNT + 1))
            shift
            ;;
        --clear-cache|-cc)
            DO_CLEAR_CACHE=true
            shift
            ;;
        --ig|--instagram)
            DO_IG=true
            shift
            ;;
        --fb|--facebook)
            DO_FB=true
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
    MODULE_COUNT=1
fi

check_env

# ── Xử lý tác vụ Web UI ─────────────────────────────────────
if [ "$DO_UI" = true ]; then
    info "🌐 Khởi chạy Face Recognition Studio Web UI tại http://localhost:${UI_PORT}..."
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "   🚀 Web UI: ${BOLD}http://localhost:${UI_PORT}${NC}"
    echo -e "   📊 Trực quan hóa Dataset, Quản lý ảnh, Nhận diện & Gom nhóm"
    echo -e "   🌐 Nhấn Ctrl+C để dừng server"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    if command -v xdg-open >/dev/null 2>&1; then
        (sleep 1.2 && xdg-open "http://localhost:${UI_PORT}" >/dev/null 2>&1) &
    fi
    PYTHONPATH="$SCRIPT_DIR" "$PYTHON" -m uvicorn facecls.server:app --host "$UI_HOST" --port "$UI_PORT" --reload
    exit 0
fi

# ── Xử lý tác vụ Download riêng ─────────────────────────────
if [ "$DO_DOWNLOAD" = true ]; then
    DL_ARGS=("--query" "$DOWNLOAD_QUERY")
    if [ "$DO_IG" = true ]; then
        DL_ARGS+=("--ig")
    fi
    if [ "$DO_FB" = true ]; then
        DL_ARGS+=("--fb")
    fi
    if [ -n "$TARGET_DIR" ]; then
        if [ "$TARGET_DIR" = "chua_phan_loai" ]; then
            DL_ARGS+=("--test")
        elif [[ "$TARGET_DIR" = /* ]]; then
            # Đường dẫn tuyệt đối (vd: /home/thang/.../download)
            DL_ARGS+=("--dest" "$TARGET_DIR")
        elif [[ "$TARGET_DIR" = ./* ]] || [[ "$TARGET_DIR" = dataset/* ]] || [[ "$TARGET_DIR" = download* ]]; then
            # Đường dẫn tương đối từ gốc dự án (vd: ./download, download, dataset/sontung)
            DL_ARGS+=("--dest" "$SCRIPT_DIR/$TARGET_DIR")
        elif [ -d "$SCRIPT_DIR/dataset/$TARGET_DIR" ]; then
            # Tên thư mục person trong dataset (vd: -d sontung)
            DL_ARGS+=("--dest" "$SCRIPT_DIR/dataset/$TARGET_DIR")
        else
            # Mặc định lưu vào dataset/<tên> nếu chỉ truyền tên người (vd: -d mono)
            DL_ARGS+=("--dest" "$SCRIPT_DIR/dataset/$TARGET_DIR")
        fi
    fi
    if [ -n "$SAMPLE_COUNT" ]; then
        DL_ARGS+=("--count" "$SAMPLE_COUNT")
    fi
    PYTHONPATH="$SCRIPT_DIR" "$PYTHON" "$DOWNLOAD" "${DL_ARGS[@]}"
    exit 0
fi

# ── Tối ưu hóa: Nếu chạy nhiều module cùng lúc -> Chạy Pipeline liên hoàn (1 process) ──
if [ "$MODULE_COUNT" -gt 1 ]; then
    info "🚀 Kích hoạt chế độ Pipeline liên hoàn (Chỉ nạp Model 1 lần duy nhất)..."
    PIPE_ARGS=()
    if [ "$DO_PREPARE" = true ]; then PIPE_ARGS+=("--prepare"); fi
    if [ "$DO_RENAME" = true ]; then PIPE_ARGS+=("--rename"); fi
    if [ "$DO_CLUSTER" = true ]; then PIPE_ARGS+=("--cluster-faces"); fi
    if [ "$DO_ORGANIZE" = true ]; then PIPE_ARGS+=("--organize"); fi
    if [ "$DO_NORMALIZE" = true ]; then PIPE_ARGS+=("--clean"); fi
    if [ "$IS_APPLY" = true ]; then PIPE_ARGS+=("--apply"); fi
    if [ "$IS_YES" = true ]; then PIPE_ARGS+=("-y"); fi
    if [ "$DO_CLEAR_CACHE" = true ]; then PIPE_ARGS+=("--clear-cache"); fi
    if [ ${#DIR_ARGS[@]} -gt 0 ]; then PIPE_ARGS+=("${DIR_ARGS[@]}"); fi

    PYTHONPATH="$SCRIPT_DIR" "$PYTHON" -m facecls.cli pipeline "${PIPE_ARGS[@]}"

    if [ "$IS_APPLY" = false ]; then
        echo -e "${YELLOW}💡 Chế độ DRY-RUN (xem thử). Để thực sự đổi tên/tạo folder/move file, thêm flag --apply${NC}"
    fi
    exit 0
fi

# ── Chạy các module đơn lẻ ──────────────────────────────────
if [ "$DO_CLEAR_CACHE" = true ]; then
    info "🗑️  Xóa cache encodings..."
    PYTHONPATH="$SCRIPT_DIR" "$PYTHON" "$MAIN" --clear-cache "${DIR_ARGS[@]}"
fi

if [ "$DO_PREPARE" = true ]; then
    info "🧪 Chuẩn bị dữ liệu kiểm thử..."
    PREP_ARGS=("${DIR_ARGS[@]}")
    if [ -n "$SAMPLE_COUNT" ]; then
        PREP_ARGS+=("--count" "$SAMPLE_COUNT")
    fi
    PYTHONPATH="$SCRIPT_DIR" "$PYTHON" "$PREPARE" "${PREP_ARGS[@]}"
fi

APPLY_FLAG=()
if [ "$IS_APPLY" = true ]; then
    APPLY_FLAG=("--apply")
fi

YES_FLAG=()
if [ "$IS_YES" = true ]; then
    YES_FLAG=("-y")
fi

if [ "$DO_RENAME" = true ]; then
    if [ "$IS_APPLY" = true ]; then
        info "✏️  Đang tiến hành ĐỔI TÊN file..."
    else
        info "🔍 Dry-run ĐỔI TÊN file..."
    fi
    PYTHONPATH="$SCRIPT_DIR" "$PYTHON" "$MAIN" "${APPLY_FLAG[@]}" "${DIR_ARGS[@]}"
    echo ""
fi

if [ "$DO_CLUSTER" = true ]; then
    if [ "$IS_APPLY" = true ]; then
        info "🤖 Đang GOM NHÓM người mới tự động..."
    else
        info "🤖 Dry-run GOM NHÓM người lạ..."
    fi
    PYTHONPATH="$SCRIPT_DIR" "$PYTHON" "$CLUSTER" "${APPLY_FLAG[@]}" "${DIR_ARGS[@]}"
    echo ""
fi

if [ "$DO_ORGANIZE" = true ]; then
    if [ "$IS_APPLY" = true ]; then
        info "🚚 Đang DI CHUYỂN các file về đúng folder người (trừ unknown_*)..."
    else
        info "🔍 Dry-run DI CHUYỂN file về folder người..."
    fi
    PYTHONPATH="$SCRIPT_DIR" "$PYTHON" "$ORGANIZE" "${APPLY_FLAG[@]}" "${YES_FLAG[@]}" "${DIR_ARGS[@]}"
    echo ""
fi

if [ "$DO_NORMALIZE" = true ]; then
    if [ "$IS_APPLY" = true ]; then
        info "🔢 Đang CHUẨN HÓA tên file trong folder (tên + STT)..."
    else
        info "🔍 Dry-run CHUẨN HÓA tên file (tên + STT)..."
    fi
    PYTHONPATH="$SCRIPT_DIR" "$PYTHON" "$NORMALIZE" "${APPLY_FLAG[@]}" "${DIR_ARGS[@]}"
    echo ""
fi

if [ "$IS_APPLY" = false ]; then
    echo -e "${YELLOW}💡 Chế độ DRY-RUN (xem thử). Để thực sự đổi tên/tạo folder/move file, thêm flag --apply${NC}"
fi
