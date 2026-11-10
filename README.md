# Face Recognition Classifier & Studio

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Face Engine](https://img.shields.io/badge/engine-InsightFace%20(ArcFace%20buffalo__l)%20%7C%20dlib-purple.svg)](https://github.com/deepinsight/insightface)
[![Web Framework](https://img.shields.io/badge/studio-FastAPI%20%2B%20TailwindCSS-emerald.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-44%20passed%20(100%25)-brightgreen.svg)](tests/)

A modern, high-performance face recognition and dataset management suite featuring **Face Recognition Studio (FastAPI Web GUI)**, an intelligent multi-platform face downloader, smart video/static photo analysis, NudeNet NSFW & sensitive content moderation, automated DBSCAN clustering, and a modular CLI pipeline.

---

## 🌟 Key Features

### 1. 🎨 Face Recognition Studio (Web GUI)
- **Live Dark-Mode Interface**: Built with FastAPI, Tailwind CSS, and Lucide icons.
- **Real-Time SSE Streaming**: Live progress bars, candidate stream previews, and detailed similarity scoring.
- **Interactive Lightbox Viewer with Bounding Boxes**: Seamlessly preview high-resolution photos, toggle detection bounding box overlays (face + NSFW sensitive regions), and play verified `.mp4` video clips directly in the browser.
- **Tabbed Operations**: Fast switching between **Thu Thập (Downloader)**, **Phân Loại (Classifier)**, and **Dataset Gallery**.

### 2. 🎯 Dual-Engine Recognition & ArcFace Embeddings
- **InsightFace (ArcFace `buffalo_l`)**: 512-dimensional deep facial representations with landmark alignment, quality filtering, and high accuracy.
- **dlib / `face_recognition`**: 128-dimensional standard embeddings.
- **Smart Quality Filtering**: Automatically ignores tiny, blurry, or low-confidence face candidates.
- **Multi-Face Resolution Strategies**: Supports `largest`, `highest_det_score`, `all`, or `skip` for images containing multiple people.

### 3. 🔞 NudeNet NSFW & Sensitive Content Moderation
- **Built-in ONNX Detection Model (`320n.onnx`)**: Fast, offline neural network inference without external API dependencies.
- **Dual Processing Modes**:
  - `rename`: Directly renames classified files with `sensitive_` prefix (e.g., `sensitive_sontung_IMG_0001.jpg`).
  - `copy`: Keeps a clean original `sontung_IMG_0001.jpg` and produces a marked duplicate copy `sensitive_sontung_IMG_0001.jpg`.
- **Shared Quarantine Directory (`dataset/sensitive/`)**: During `organize` (file relocation), all sensitive photos and videos are automatically gathered into a shared `dataset/sensitive/` folder directly, keeping person reference folders clean.
- **Strictly Excluded from AI Models**: The `sensitive` directory is registered in `exclude_dirs`, ensuring quarantined files are never accidentally trained as facial reference embeddings.
- **Preserved During Normalization (`clean`)**: The canonical numbering tool preserves `sensitive_` prefixes (`sensitive_<person>_<index>.<ext>`).

### 4. 👁️ Interactive Bounding Box Overlays (Khoanh vùng nhận diện)
- **Surgical HUD Reticles**: Live bounding boxes for recognized faces (Cyan `#06b6d4`), low-confidence matches (Amber `#f59e0b`), and unknown strangers (Rose `#f43f5e`).
- **NudeNet Sensitive Regions**: Color-coded overlay boxes with Vietnamese labels and confidence percentages (e.g. `🔞 Ngực nữ (Bikini/Đồ lót): 89.2%`, `Vùng kín: 95.1%`).
- **One-Click Toggle Button**: Easily show/hide all bounding boxes with the `[👁️ Khoanh vùng: BẬT/TẮT]` button or keyboard shortcut `B`.
- **Interactive Detection Chips**: Tag list at the bottom of the lightbox modal; clicking any tag highlights that specific bounding box.
- **Zoom-Synchronized Reticles**: Zooming (`+`/`-`) scales both the image and the bounding boxes simultaneously with pixel-perfect alignment.

### 5. 🌐 Multi-Source Intelligent Downloader & Candidate Saver
- **Bing Web Search**: Infinite pagination with same-person identity verification.
- **Instagram Profiles & Highlights**: Crawls profile avatars, high-resolution post photos, multi-image sidecars (carousels), Reels, and story highlights using authenticated browser session cookies.
- **Facebook Albums & Profiles**: Stream photos and profile pictures via graph scraper and authenticated browser cookies.
- **Anchor Centroid Verification**: Dynamically updates an ArcFace anchor representation with a moving average to ensure all downloaded media belongs strictly to the targeted individual.
- **Manual Failed Candidate Saving**: Inspect any rejected/borderline candidate in the lightbox preview and click **"Lưu ảnh này" (Save Candidate)** to manually override and save it directly into the dataset.

### 6. 🎬 Smart Video Face Recognition & Static Photo Detection
- **Video Face Recognition (Majority Voting)**: Extracts sample frames, computes embeddings with motion-blur adjusted thresholds (`min(det_thresh, 0.42)`), and votes on candidate identities across both CLI and Web Studio (`/api/scan`, `/api/scan/stream`, `/api/playground`).
- **Resilient Video Frame Decoding**: Sequential fallback ensures reliable frame extraction even for streaming, web, or variable frame-rate MP4s where `total_frames <= 0`.
- **Single-Photo Video Optimization**: Intelligently identifies videos that only display a single static photo (e.g. still pictures with background audio, lyrics, or subtle zoom) using 3-tier motion analysis (Mean Absolute Difference + $8 \times 8$ grid block variance + facial landmark consistency). Skips saving heavy `.mp4` files and extracts 1 crisp `.jpg` frame.
- **Video Sensitive Checking**: Samples video frames and detects sensitive content in clips.

### 7. 🤖 Automated Pipeline & Dataset Curation
- **DBSCAN Auto-Clustering**: Automatically groups unidentified faces of new recurring individuals.
- **Interactive Registration**: Prompts for real human names and moves files into organized subdirectories.
- **Dataset File Normalization**: Formats files into canonical names: `<person_name>_<index>.<ext>`.
- **Persistent Embedding Cache**: MD5-fingerprinted cache (`.face_encodings_cache.pkl`) guarantees instant startup and prevents redundant neural network inferences.

---

## 📁 Project Architecture

```text
face-recognition-classifier/
├── facecls/                       # Core Python Package
│   ├── __init__.py                # Package exports
│   ├── cache.py                   # Thread-safe persistent embedding cache
│   ├── cli.py                     # Unified Click CLI commands
│   ├── config.py                  # YAML application configuration loader
│   ├── downloader.py              # Web, Instagram & Facebook scrapers
│   ├── engine.py                  # InsightFace & dlib wrapper with quality filtering
│   ├── io.py                      # Safe file IO, renaming, moving & normalization
│   ├── matcher.py                 # Cosine similarity matcher & DBSCAN clustering
│   ├── pipeline.py                # End-to-end classification & organization pipelines
│   ├── sensitive.py               # NudeNet NSFW & sensitive content detector
│   ├── server.py                  # FastAPI server with SSE streaming endpoints
│   ├── templates/
│   │   └── index.html             # Studio Web GUI application
│   └── video.py                   # Video stream extraction & static motion analysis
├── tests/                         # Pytest Suite (44 tests)
│   ├── test_cache.py              # Cache persistence and invalidation tests
│   ├── test_config.py             # Config loading and validation tests
│   ├── test_io.py                 # Safe rename, move, and filename builder tests
│   ├── test_matcher.py            # Matching and clustering tests
│   ├── test_sensitive.py          # NudeNet, sensitive naming, organize & clean tests
│   ├── test_server.py             # FastAPI API endpoint & video tests
│   └── test_video.py              # Video URL, sampling, and static video detection tests
├── config.yaml                    # Global configuration file
├── download_faces.py              # Standalone Downloader CLI
├── rename_faces.py                # Standalone Renaming CLI
├── cluster_faces.py               # Standalone DBSCAN Clustering CLI
├── organize_faces.py              # Standalone File Organization CLI
├── clean_names.py                 # Standalone Filename Normalizer CLI
├── prepare_test_data.py           # Dataset sampling & test preparation CLI
├── requirements.txt               # Python package dependencies
├── run.sh                         # Unified Bash CLI runner
└── dataset/                       # Dataset directory (ignored by git)
    ├── person_a/                  # Reference photos for Person A
    ├── person_b/                  # Reference photos for Person B
    ├── sensitive/                 # Quarantined NSFW / sensitive media (shared)
    └── chua_phan_loai/            # Unclassified input images & videos
```

---

## ⚡ Quick Start

### 1. Installation & Environment Setup

Automatic setup creates a Python virtual environment and installs all dependencies:

```bash
chmod +x run.sh
./run.sh --setup
```

### 2. Launching Face Recognition Studio (Web GUI)

Start the local server and open the web studio:

```bash
./run.sh --ui
# Shorthand aliases: ./run.sh -u or ./run.sh --studio
# Or manually with uvicorn:
# PYTHONPATH=. .venv/bin/uvicorn facecls.server:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at `http://localhost:8000/`.

---

## 🚀 CLI Usage & Shorthands

The runner script `run.sh` provides shorthands for single tasks, chained pipelines, and batch operations:

### Web Studio & Downloader
```bash
# Launch Web GUI Studio
./run.sh --ui          # Or: ./run.sh -u / ./run.sh --studio

# Download faces from Web / Social Media
./run.sh --download "sontung" -d sontung --count 20
```

### Pipeline & Classification
```bash
# Preview face recognition (Dry-Run)
./run.sh -r

# Execute face recognition renaming
./run.sh -ra

# Rename with NSFW sensitive detection in 'rename' mode
python -m facecls rename --apply --sensitive-mode rename --sensitive-thresh 0.35

# Rename with NSFW sensitive detection in 'copy' mode (keeps clean + duplicate sensitive)
python -m facecls rename --apply --sensitive-mode copy --sensitive-thresh 0.35

# Auto-cluster unknown faces in chua_phan_loai
./run.sh -c

# Full automated pipeline: Cluster + Rename + Organize
./run.sh -cao

# Non-interactive mode (auto-accept default cluster names)
./run.sh -caoy

# Run pipeline on a custom directory
./run.sh -cao -d other

# Standardize filenames inside a person folder to <name>_<index>.<ext>
./run.sh -na -d binhan
```

### CLI Flag Reference

| Short | Full Flag | Description |
|---|---|---|
| `-u` / `-s` | `--ui` / `--studio` | Launch Face Recognition Studio Web UI |
| `--download` | `--download <query>` | Collect face images from Web / Instagram / Facebook |
| `-a` | `--apply` | Execute mutative changes (rename, move, create directories) |
| `-r` | `--rename` | Match and rename unclassified images against known persons |
| `--sensitive-mode` | `--sensitive-mode <none\|rename\|copy>` | NSFW detection mode (`rename` or duplicate `copy`) |
| `--sensitive-thresh` | `--sensitive-thresh <float>` | NudeNet confidence score threshold (default: 0.35) |
| `-c` | `--cluster` | Auto-cluster unknown faces using DBSCAN |
| `-o` / `-m` | `--organize` | Move classified files into dedicated person folders & `dataset/sensitive/` |
| `-n` | `--clean` / `--norm` | Standardize dataset filenames to `<person>_<index>.<ext>` |
| `-y` | `--yes` | Non-interactive auto-confirmation for prompts |
| `-p` | `--prepare-test` | Sample test images from person folders into unclassified dir |
| `-d <dir>` | `--dir <dir>` | Specify target directory |
| `-cc` | `--clear-cache` | Clear persistent face embedding cache |

---

## ⚙️ Configuration (`config.yaml`)

```yaml
dataset_root: "./dataset"
unclassified_dir: "chua_phan_loai"
cache_file: ".face_encodings_cache.pkl"

# Folders completely excluded from face recognition models & training
exclude_dirs:
  - "chua_phan_loai"
  - "khac"
  - "other"
  - "sensitive"        # Quarantined NSFW media (never trained)
  - "GV"
  - "DD"

# Embedding engine: insightface (recommended) or face_recognition (dlib)
engine: "insightface"
insightface_model: "buffalo_l"

# Multi-face resolution: largest, highest_det_score, all, skip
multi_face_strategy: "largest"
det_thresh: 0.50

# Similarity threshold (0.0 - 1.0; lower is stricter)
tolerance: 0.55

# DBSCAN clustering threshold
cluster_eps: 0.38
cluster_min_samples: 2

# Maximum images sampled per person
max_images_per_person: 50

# Confidence threshold (%) to trigger low_ prefix
low_confidence_threshold: 70

# NudeNet NSFW & Sensitive Content Moderation
enable_sensitive_check: false
sensitive_mode: "rename"       # "rename" or "copy"
sensitive_threshold: 0.35      # 0.0 - 1.0; recommended: 0.35
sensitive_prefix: "sensitive"
```

---

## 🧪 Testing

Run the automated test suite with `pytest`:

```bash
PYTHONPATH=. .venv/bin/pytest tests/
```

All 44 test cases verify cache persistence, configuration loading, IO safety, face matching, REST/SSE endpoints, NudeNet sensitive checks, lightbox reticles, video recognition, and video motion analysis.

---

## 📄 License

This project is licensed under the MIT License.
