# Face Recognition Classifier & Studio

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Face Engine](https://img.shields.io/badge/engine-InsightFace%20(ArcFace%20buffalo__l)%20%7C%20dlib-purple.svg)](https://github.com/deepinsight/insightface)
[![Web Framework](https://img.shields.io/badge/studio-FastAPI%20%2B%20TailwindCSS-emerald.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-33%20passed%20(100%25)-brightgreen.svg)](tests/)

A modern, high-performance face recognition and dataset management suite featuring **Face Recognition Studio (FastAPI Web GUI)**, an intelligent multi-platform face downloader, smart video/static photo analysis, automated DBSCAN clustering, and a modular CLI pipeline.

---

## 🌟 Key Features

### 1. 🎨 Face Recognition Studio (Web GUI)
- **Live Dark-Mode Interface**: Built with FastAPI, Tailwind CSS, and Lucide icons.
- **Real-Time SSE Streaming**: Live progress bars, candidate stream previews, and detailed similarity scoring.
- **Interactive Lightbox Viewer**: Seamlessly preview high-resolution photos and play verified `.mp4` video clips directly in the browser.
- **Tabbed Operations**: Fast switching between **Thu Thập (Downloader)**, **Phân Loại (Classifier)**, and **Dataset Gallery**.

### 2. 🎯 Dual-Engine Recognition & ArcFace Embeddings
- **InsightFace (ArcFace `buffalo_l`)**: 512-dimensional deep facial representations with landmark alignment, quality filtering, and high accuracy.
- **dlib / `face_recognition`**: 128-dimensional standard embeddings.
- **Smart Quality Filtering**: Automatically ignores tiny, blurry, or low-confidence face candidates.
- **Multi-Face Resolution Strategies**: Supports `largest`, `highest_det_score`, `all`, or `skip` for images containing multiple people.

### 3. 🌐 Multi-Source Intelligent Downloader
- **Bing Web Search**: Infinite pagination with same-person identity verification.
- **Instagram Profiles**: Crawls profile avatars, high-resolution post photos, multi-image sidecars (carousels), and Reels.
- **Instagram Highlights**: Fetches private/public story highlights using authenticated browser session cookies.
- **Facebook Albums & Profiles**: Stream photos and profile pictures via graph scraper and authenticated browser cookies.
- **Anchor Centroid Verification**: Dynamically updates an ArcFace anchor representation with a moving average to ensure all downloaded media belongs strictly to the targeted individual.

### 4. 🎬 Smart Video Analysis & Static Photo Detection
- **In-Memory Frame Sampling**: Extracts frames at regular time intervals without disk bloat, verifying face identity against the anchor centroid.
- **Single-Photo Video Optimization**: Intelligently identifies videos that only display a single static photo (e.g. still pictures with background audio, lyrics, or subtle zoom) using 3-tier motion analysis (Mean Absolute Difference + $8 \times 8$ grid block variance + facial landmark consistency). Skips saving heavy `.mp4` files and extracts 1 crisp `.jpg` frame.
- **Direct Video Stream Processing**: Supports direct CDN links, Reels, Shorts, and `.mp4` URLs.

### 5. 🤖 Automated Pipeline & Dataset Curation
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
│   ├── server.py                  # FastAPI server with SSE streaming endpoints
│   ├── templates/
│   │   └── index.html             # Studio Web GUI application
│   └── video.py                   # Video stream extraction & static motion analysis
├── tests/                         # Pytest Suite (33 tests)
│   ├── test_cache.py              # Cache persistence and invalidation tests
│   ├── test_config.py             # Config loading and validation tests
│   ├── test_io.py                 # Safe rename, move, and filename builder tests
│   ├── test_matcher.py            # Matching and clustering tests
│   ├── test_server.py             # FastAPI API endpoint tests
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
| `-c` | `--cluster` | Auto-cluster unknown faces using DBSCAN |
| `-o` / `-m` | `--organize` | Move classified files into dedicated person folders |
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
```

---

## 🧪 Testing

Run the automated test suite with `pytest`:

```bash
PYTHONPATH=. .venv/bin/pytest tests/
```

All 33 test cases verify cache persistence, configuration loading, IO safety, face matching, REST/SSE endpoints, and video motion analysis.

---

## 📄 License

This project is licensed under the MIT License.
