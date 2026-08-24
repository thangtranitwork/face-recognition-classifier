# Face Recognition File Renamer

An automated CLI tool that detects faces in unclassified images and renames files based on matched persons using deep learning (`dlib` / `face_recognition`).

---

## 📁 Directory Structure

Organize your dataset folder as shown below:

```text
face-recognition-classifier/
├── rename_faces.py       # Main Python script
├── config.yaml           # Configuration file
├── run.sh                # Executable runner script
├── requirements.txt      # Python dependencies
└── dataset/              # Your dataset directory (ignored by git)
    ├── person_A/         # Classifiers for Person A (labeled images)
    ├── person_B/         # Classifiers for Person B
    ├── other/            # Excluded directory (configurable)
    └── chua_phan_loai/   # Target directory containing unclassified images
```

---

## ⚡ Quick Start

### 1. Installation

Automatic setup (creates virtual environment, installs dependencies, and applies compatibility patches):

```bash
chmod +x run.sh
./run.sh --setup
```

*Or manually install system dependencies & python requirements:*

```bash
# Ubuntu / Debian
sudo apt-get install -y cmake libopenblas-dev liblapack-dev

# Virtual Environment & Requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### Preview Mode (Dry-Run)
Inspect recognition results and proposed file renaming **without modifying any files**:

```bash
./run.sh
# Or directly via Python:
.venv/bin/python rename_faces.py
```

### Apply Renaming
Execute the actual file renaming:

```bash
./run.sh --apply
# Or directly via Python:
.venv/bin/python rename_faces.py --apply
```

### Rebuild Face Cache
Invalidate cached encodings when adding new training images or folders:

```bash
./run.sh --clear-cache
```

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

# Maximum sample images encoded per person (optimizes speed for large datasets)
max_images_per_person: 50

# Confidence threshold (%) to trigger low_ prefix
low_confidence_threshold: 70
```

---

## ✨ Features

- **Smart Encoding Cache**: MD5-fingerprinted cache ensures instant startup after initial encoding.
- **Adaptive Face Sampling**: Evaluates images until target valid face encodings are acquired, skipping images without faces.
- **Dry-Run Default**: Safe execution mode prevents accidental mass file renaming.
- **Detailed Logging**: Logs match results, confidence levels, and distance metrics to `rename_log.txt`.
