# Face Recognition File Renamer

An automated CLI tool that detects faces in unclassified images/videos, auto-clusters unknown faces (DBSCAN), renames files based on matched persons using deep learning (`dlib` / `face_recognition`), and organizes dataset folders.

---

## 📁 Directory Structure

Organize your dataset folder as shown below:

```text
face-recognition-classifier/
├── rename_faces.py       # Main face recognition & renaming script
├── cluster_faces.py      # Auto-clustering unknown faces (DBSCAN)
├── organize_faces.py     # Move renamed files & prompt for new person names
├── clean_names.py        # Normalize dataset file names to <person_name>_<stt>
├── prepare_test_data.py  # Reset & prepare test dataset
├── config.yaml           # Configuration file
├── run.sh                # Flexible CLI runner script
├── requirements.txt      # Python dependencies
└── dataset/              # Dataset directory (ignored by git)
    ├── binhan/           # Training folder for Person A
    ├── sontung/          # Training folder for Person B
    ├── other/            # Custom/excluded directory
    └── chua_phan_loai/   # Unclassified images/videos
```

---

## ⚡ Quick Start

### 1. Installation

Automatic setup (creates virtual environment, installs dependencies, and applies compatibility patches):

```bash
chmod +x run.sh
./run.sh --setup
```

---

## 🚀 Flexible CLI & Short Flags

The `run.sh` script supports single-letter shorthands, combined flags, and custom target directories:

```bash
# Preview face recognition (Dry-Run)
./run.sh

# Apply face recognition renaming
./run.sh -a

# Auto-cluster unknown faces in chua_phan_loai (Dry-Run)
./run.sh -c

# Auto-cluster unknown faces and rename files in place
./run.sh -ca

# Full workflow: Cluster + Apply + Organize (asks for real person names!)
./run.sh -cao

# Auto-accept default names during organize without prompt (-y)
./run.sh -caoy

# Run on a custom folder (e.g. 'other' or '/path/to/dir')
./run.sh -cao -d other

# Normalize file names inside person folders to <name>_<STT> (e.g. binhan_1.jpg, binhan_2.png)
./run.sh -na
./run.sh -na -d binhan
```

### 📋 Short Flags Reference

| Short | Full Flag | Description |
|---|---|---|
| `-a` | `--apply` | Execute changes (rename, move, create folder) |
| `-r` | `--rename` | Face recognition & renaming |
| `-c` | `--cluster` | Auto-cluster unknown faces (DBSCAN) |
| `-o` / `-m` | `--organize` | Move classified files & prompt for new person names |
| `-n` | `--clean` / `--norm` | Standardize dataset filenames to `<person>_<index>` |
| `-y` | `--yes` | Non-interactive auto-accept default names |
| `-p` | `--prepare-test` | Reset dataset & sample test images into `chua_phan_loai/` |
| `-d <dir>` | `--dir <dir>` | Specify target directory (`other`, `chua_phan_loai`, etc.) |
| `-cc` | `--clear-cache` | Clear face encoding cache |

---

## 🏷️ Renaming Rules & Confidence Tiers

The tool renames files according to face matching confidence:

| Scenario | Input Example | Output Example | Log Status |
|---|---|---|---|
| **High Confidence** ($\ge 70\%$) | `IMG_001.jpg` | `person_A_IMG_001.jpg` | ✅ `[confidence 100.0%]` |
| **Low Confidence** ($< 70\%$) | `IMG_002.jpg` | `low_person_A_IMG_002.jpg` | ⚠️ `[confidence 54.2%]` |
| **No Face Detected** | `IMG_003.jpg` | `unknown_IMG_003.jpg` | ❓ `[no face detected]` |
| **Multiple Faces** | `IMG_004.jpg` | `unknown_IMG_004.jpg` | ❓ `[multiple faces detected]` |

---

## ⚙️ Configuration (`config.yaml`)

Key options available in `config.yaml`:

```yaml
dataset_root: "./dataset"
unclassified_dir: "chua_phan_loai"

# Exclude non-person folders from encoding
exclude_dirs:
  - "chua_phan_loai"
  - "khac"
  - "other"

# Recognition threshold (0.0 - 1.0; lower is stricter)
tolerance: 0.55

# DBSCAN clustering threshold (0.35 - 0.40)
cluster_eps: 0.38
cluster_min_samples: 2

# Maximum sample images encoded per person
max_images_per_person: 50

# Confidence threshold (%) to trigger low_ prefix
low_confidence_threshold: 70
```

---

## ✨ Features

- **Video Recognition Support**: Frame sampling + majority voting for `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.
- **DBSCAN Auto-Clustering**: Groups unidentified faces of new persons appearing multiple times.
- **Interactive Person Registration**: Prompts user for real names when moving clustered files.
- **Filename Normalization**: Standardizes dataset files to `<person>_<1,2,3>.<ext>`.
- **Smart Encoding Cache**: MD5-fingerprinted cache ensures instant startup after initial encoding.
- **Dry-Run Default**: Safe execution mode prevents accidental mass file renaming.
