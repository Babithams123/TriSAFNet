# TriSAFNet — Karnataka LULC Classification

**TriSAFNet** (*Tri-Source Attention Fusion Network*) is a multi-branch deep learning model for land-use / land-cover (LULC) mapping over Karnataka, India. It fuses **SAR**, **optical / spectral**, and **topographic** inputs with attention mechanisms (SE, CBAM, cross-source attention) and is compared against CNN, CNN–RF, EfficientNet, MobileNet, ViT, and classical ML baselines.

This repository contains the **public experiment code** used for the manuscript. **Training rasters, patches, and labels are not included.**

---

## Data availability

Rasters, patch archives, and labeled training points are available from the corresponding author on reasonable request:

**Email:** `mehathab.ashraff@gmail.com` or `babithabms@gmail.com`

After you receive the data package, place files under `./data/` as described below (see *Expected data layout*).

---

## Repository layout

```
LULC - TriFANet/
├── configs/
│   └── config.py              # Hyperparameters, band layout, paths
├── scripts/
│   ├── 01_gee_export.js       # Google Earth Engine export
│   ├── 02_data_pipeline.py    # Patch extraction + spatial CV prep
│   ├── 03_run_experiments.py  # Train / evaluate all models
│   ├── 04_generate_maps.py    # Full-area map inference
│   ├── 05_train_trisafnet.py  # TriSAFNet-only CV training
│   ├── 06_generate_maps_trisafnet.py
│   ├── 07_publication_maps.py
│   ├── 08_geemap_publication_maps.py
│   ├── models.py              # TriSAFNet + baselines
│   ├── run_all.py             # End-to-end orchestrator
│   └── …                      # GEE helpers, utilities
├── utils/
│   └── mlflow_utils.py
├── data/                      # Empty — request data from author
├── outputs/                   # Created at runtime
├── requirements.txt
└── README.md
```

---

## Requirements

- Python **3.9–3.10** recommended  
- NVIDIA GPU + CUDA strongly recommended for deep models  
- Dependencies: see `requirements.txt`

```bash
conda create -n trisafnet python=3.10 -y
conda activate trisafnet
pip install -r requirements.txt
# Optional GPU TensorFlow (match your CUDA stack):
# pip install "tensorflow[and-cuda]"
```

---

## Expected data layout

Once you have the author data package, `./data/` should look like:

```
data/
├── stack_2018.tif                 # Multi-band study-area stack
├── stack_2024.tif
├── training_points_2018.csv       # Labeled samples
├── training_points_2024.csv
└── scaler_stats_{7x7,15x15,21x21}.npz   # Optional (or rebuild)
```

Paths and band order are defined in `configs/config.py` (`N_BANDS = 27` by default: C-/L-band SAR, Sentinel-2 reflectance + indices, LST, elevation, slope).

You may also use pre-built patch archives if provided:

```
data/patches_{2018|2024}_{7x7|15x15|21x21}.npz
data/patches_{2018|2024}_{7x7|15x15|21x21}_noaug.npz
```

---

## Quick start

### 1. Export / prepare data (optional if author package already has stacks)

1. Open [Google Earth Engine Code Editor](https://code.earthengine.google.com/)
2. Run `scripts/01_gee_export.js` (adapt asset paths / training collections)
3. Download exports into `./data/`

### 2. Build patches

```bash
python scripts/02_data_pipeline.py --year both --patch-size 21
```

### 3. Run experiments

```bash
# Full comparison for one year
python scripts/03_run_experiments.py --year 2018

# Include TriSAFNet component ablation
python scripts/03_run_experiments.py --year 2018 --ablation-trisafnet

# Both years
python scripts/03_run_experiments.py --year both
```

### 4. Or run the full pipeline

```bash
python scripts/run_all.py --years 2018 2024 --patch-sizes 21 --no-sleep
```

Useful flags: `--skip-data`, `--skip-ablation`, `--skip-maps`, `--no-mlflow`.

### 5. Train TriSAFNet only / map generation

```bash
python scripts/05_train_trisafnet.py --year both
python scripts/06_generate_maps_trisafnet.py --year both
python scripts/04_generate_maps.py --patch-size 21
```

Results land under `./outputs/` and `./runs/` (gitignored).

---

## Models

| Role | Model |
|------|--------|
| Proposed | **TriSAFNet** (SAR / optical / topo branches + SE + CBAM + fusion) |
| Baselines | Lightweight CNN–RF, CNN-only, CNN–LightGBM, MobileNetV2, EfficientNet-B0, Compact ViT, RF, SVM, ANN, XGBoost, LightGBM, … |

Default patch size is **21×21** (`PATCH_SIZE` in `configs/config.py`). Ablation sizes: 7, 15, 21.

---

## Citation

If you use this code, please cite the associated TriSAFNet: A Tri - Source Attention Fusion Network for Multi-Sensor Land Use and Land Cover Classification in Urban–Peri-Urban Karnataka:

```
M.S. Babitha1, Diana Andrushia2, A. Mehathab3, N. Anand4,  M.Z. Naser. TriSAFNet: A Tri - Source Attention Fusion Network for Multi-Sensor Land Use and Land Cover Classification in Urban–Peri-Urban Karnataka, 2026.
Manuscript ID: RSASE-D-25-02735
```

---

## License / contact

Code is provided for research reproducibility related to the manuscript.  
For data access or questions: **`mehathab.ashraff@gmail.com`** or **`babithabms@gmail.com`**

---

## Notes

- Large rasters and `.npz` patch files are intentionally **excluded** from GitHub (see `.gitignore`).
- Do not commit local `outputs/`, `runs/`, or `mlruns/` directories.
- Hyperparameters and paths live in a single place: `configs/config.py`.
