"""
=============================================================================
  COLAB NOTEBOOK: Generate LULC Classification Maps
  CNN-RF Pipeline — Multi-Model Comparison
=============================================================================

Copy each section (separated by # %% markers) as a separate Colab cell.
Or upload this entire file and run it as-is.

Workflow:
  1. Mount Google Drive
  2. Install dependencies
  3. Load models + scaler
  4. Classify tiles (memory-efficient, strip-by-strip)
  5. Render publication-quality comparison maps

Required Google Drive structure:
  My Drive/
  └── CNN_RF_Pipeline/
      ├── models_ps7/            ← model files for patch_size=7
      ├── scaler_stats_7x7.npz   ← scaler file (tiny)
      ├── CNN_RF_Stack2018_Data/  ← GeoTIFF tiles
      └── CNN_RF_Stack2024_Data/  ← GeoTIFF tiles
"""

# %% [markdown]
# # 🗺️ LULC Classification Map Generation
# Run each cell in order. Tiles that are already classified will be skipped automatically.

# %% ============================================================
# CELL 1: Mount Drive & Install Dependencies
# ============================================================
import os, sys

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# IMPORTANT: Set legacy Keras BEFORE any TF import
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Install geospatial packages + tf_keras for compatibility
get_ipython().system('pip install -q rasterio joblib lightgbm xgboost tf_keras')

import tensorflow as tf
print(f"✅ TensorFlow {tf.__version__}")
print(f"   TF_USE_LEGACY_KERAS = {os.environ.get('TF_USE_LEGACY_KERAS', 'not set')}")

# %% ============================================================
# CELL 2: Configuration
# ============================================================
import numpy as np
import gc
import json
import time
from pathlib import Path

# ---- PATHS (adjust if your Drive folder names differ) ----
DRIVE_BASE = "/content/drive/MyDrive/CNN_RF_Pipeline"

STACK_2018_DIR = f"{DRIVE_BASE}/CNN_RF_Stack2018_Data"
STACK_2024_DIR = f"{DRIVE_BASE}/CNN_RF_Stack2024_Data"
MODELS_DIR     = f"{DRIVE_BASE}/models_ps7"
SCALER_PATH    = f"{DRIVE_BASE}/scaler_stats_7x7.npz"
OUTPUT_DIR     = f"{DRIVE_BASE}/map_outputs"

# ---- CLASSIFICATION CONFIG ----
PATCH_SIZE = 7
N_BANDS    = 27
N_CLASSES  = 5
STRIP_ROWS = 128          # rows per strip — lower if OOM (64 or 32)
BATCH_SIZE = 2048          # GPU batch size for Keras models

CLASS_NAMES  = ["Forest", "Plantation", "Fallow Land", "Water Bodies", "Built-up Area"]
CLASS_COLORS = ["#00770c", "#ba02d6", "#815756", "#0821ff", "#ff0000"]

# Band index ranges for multi-branch TriSAFNet
SAR_BANDS     = list(range(0, 5))
OPTICAL_BANDS = list(range(5, 25))
TOPO_BANDS    = list(range(25, 27))

# ---- MODELS TO CLASSIFY (top baselines + proposed) ----
# Each entry: (display_name, model_type, file(s))
#   model_type: 'keras'  → single .keras file, argmax(predict)
#               'hybrid' → .keras CNN backbone + .joblib ML classifier
#               'ml'     → .joblib only, flattens patches

MODELS_CONFIG = [
    ("TriSAFNet (Proposed)", "keras",  ["TriSAFNet_Proposed.keras"]),
    ("EfficientNet-B0",    "keras",  ["EfficientNet-B0.keras"]),
    ("CNN Only",           "keras",  ["CNN_Only.keras"]),
    ("Lightweight CNN-RF", "hybrid", ["Lightweight_CNN-RF_cnn_backbone.keras",
                                      "Lightweight_CNN-RF_rf.joblib"]),
    # Uncomment any of these if needed:
    # ("RF Only",          "ml",    ["RF_Only.joblib"]),
    # ("XGBoost Only",     "ml",    ["XGBoost_Only.joblib"]),
    # ("CNN-LightGBM",     "hybrid",["CNN-LightGBM_cnn_backbone.keras",
    #                                "CNN-LightGBM_lgbm.joblib"]),
]

# ---- YEARS TO PROCESS ----
YEARS = [2018, 2024]

# Verify paths exist
print("📂 Checking Drive paths...\n")
for p in [STACK_2018_DIR, STACK_2024_DIR, MODELS_DIR, SCALER_PATH]:
    if os.path.exists(p):
        print(f"  ✅ {os.path.basename(p)}")
    else:
        print(f"  ❌ NOT FOUND: {p}")

# --- Auto-detect models directory ---
# If models_ps7/ doesn't exist, look for common alternatives
if not os.path.exists(MODELS_DIR):
    print(f"\n⚠️  Models directory not found at: {MODELS_DIR}")
    print(f"   Searching for model files in {DRIVE_BASE}...\n")
    # List what's actually in DRIVE_BASE
    if os.path.exists(DRIVE_BASE):
        print(f"   Contents of {DRIVE_BASE}:")
        for item in sorted(os.listdir(DRIVE_BASE)):
            full = os.path.join(DRIVE_BASE, item)
            tag = "📁" if os.path.isdir(full) else "📄"
            print(f"     {tag} {item}")
        # Try common folder names
        for candidate in ["models_ps7", "models", "outputs_ps7/models",
                          "outputs/models", "ps7_models"]:
            test = os.path.join(DRIVE_BASE, candidate)
            if os.path.exists(test):
                MODELS_DIR = test
                print(f"\n   ✅ Auto-detected: {MODELS_DIR}")
                break
    else:
        print(f"\n   ❌ Base directory not found: {DRIVE_BASE}")
        print(f"   📋 Your Drive root contains:")
        drive_root = "/content/drive/MyDrive"
        if os.path.exists(drive_root):
            for item in sorted(os.listdir(drive_root))[:30]:
                print(f"      {item}")

# --- List available model files ---
print(f"\n📋 Model files in {MODELS_DIR}:")
if os.path.exists(MODELS_DIR):
    for f in sorted(os.listdir(MODELS_DIR)):
        size = os.path.getsize(os.path.join(MODELS_DIR, f))
        print(f"  {'✅' if size > 1000 else '⚠️'} {f} ({size/1024:.0f} KB)")
else:
    print("  ❌ Directory does not exist!")
    print(f"\n  💡 FIX: Upload model files to Google Drive at:")
    print(f"     {MODELS_DIR}/")
    print(f"     Or update MODELS_DIR in Cell 2 to the correct path.")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nOutputs will be saved to: {OUTPUT_DIR}")

# %% ============================================================
# CELL 3: Load Scaler
# ============================================================
print("Loading scaler statistics...")
scaler_data = np.load(SCALER_PATH)
scaler_min = scaler_data['scaler_min'].astype(np.float32)
scaler_max = scaler_data['scaler_max'].astype(np.float32)
scaler_range = (scaler_max - scaler_min).astype(np.float32)
scaler_range[scaler_range == 0] = 1.0

print(f"  Bands: {len(scaler_min)}")
print(f"  Min range: [{scaler_min.min():.4f}, {scaler_min.max():.4f}]")
print(f"  Max range: [{scaler_max.min():.4f}, {scaler_max.max():.4f}]")
print("✅ Scaler loaded")

# %% ============================================================
# CELL 4: Model Wrappers
# ============================================================
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import joblib

# Enable GPU memory growth
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

print(f"GPU: {tf.config.list_physical_devices('GPU')}")

import shutil
LOCAL_MODEL_CACHE = '/tmp/cnn_rf_models'
os.makedirs(LOCAL_MODEL_CACHE, exist_ok=True)

def _copy_to_local(drive_path):
    """Copy file from Drive to /tmp for reliable loading."""
    local_path = os.path.join(LOCAL_MODEL_CACHE, os.path.basename(drive_path))
    if not os.path.exists(local_path):
        print(f"    Copying to local: {os.path.basename(drive_path)}")
        shutil.copy2(drive_path, local_path)
    return local_path


def _load_keras_robust(model_path):
    """Try multiple methods to load a .keras model file."""
    import zipfile

    # Diagnostic
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File does not exist: {model_path}")
    size = os.path.getsize(model_path)
    is_zip = zipfile.is_zipfile(model_path)
    print(f"    File: {os.path.basename(model_path)} ({size/1024:.0f} KB, zip={is_zip})")

    # Method 1: Standard load
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e1:
        print(f"    Method 1 (tf.keras) failed: {e1.__class__.__name__}")

    # Method 2: tf_keras legacy package
    try:
        import tf_keras
        return tf_keras.models.load_model(model_path, compile=False)
    except Exception as e2:
        print(f"    Method 2 (tf_keras) failed: {e2.__class__.__name__}")

    # Method 3: Rename to .h5 and try H5 format
    try:
        h5_path = model_path.replace('.keras', '.h5')
        shutil.copy2(model_path, h5_path)
        return tf.keras.models.load_model(h5_path, compile=False)
    except Exception as e3:
        print(f"    Method 3 (H5 rename) failed: {e3.__class__.__name__}")

    # Method 4: Extract zip and load config
    if is_zip:
        try:
            extract_dir = model_path + '_extracted'
            with zipfile.ZipFile(model_path, 'r') as z:
                z.extractall(extract_dir)
            return tf.keras.models.load_model(extract_dir, compile=False)
        except Exception as e4:
            print(f"    Method 4 (extract zip) failed: {e4.__class__.__name__}")

    raise RuntimeError(
        f"Cannot load {model_path}. All methods failed.\n"
        f"  File size: {size} bytes, is_zip: {is_zip}\n"
        f"  TF version: {tf.__version__}\n"
        f"  Try: restart runtime and ensure Cell 1 runs FIRST"
    )


class KerasModelWrapper:
    """Wraps a pure Keras model (.keras file) for patch-based classification."""

    def __init__(self, model_path, name=""):
        self.name = name
        print(f"  Loading Keras model: {name}")
        self.model = _load_keras_robust(model_path)
        print(f"    ✅ Loaded! Parameters: {self.model.count_params():,}")

    def predict_batch(self, patches):
        """patches: (N, H, W, C) → labels (N,)"""
        proba = self.model.predict(patches, batch_size=BATCH_SIZE, verbose=0)
        return np.argmax(proba, axis=1)

    def cleanup(self):
        del self.model
        tf.keras.backend.clear_session()
        gc.collect()


class HybridModelWrapper:
    """Wraps a CNN backbone (.keras) + ML classifier (.joblib)."""

    def __init__(self, cnn_path, ml_path, name=""):
        self.name = name
        print(f"  Loading hybrid model: {name}")
        self.cnn = tf.keras.models.load_model(cnn_path)
        self.ml = joblib.load(ml_path)
        print(f"    CNN params: {self.cnn.count_params():,}")
        print(f"    ML classifier: {type(self.ml).__name__}")

    def predict_batch(self, patches):
        """patches: (N, H, W, C) → labels (N,)"""
        features = self.cnn.predict(patches, batch_size=BATCH_SIZE, verbose=0)
        return self.ml.predict(features)

    def cleanup(self):
        del self.cnn, self.ml
        tf.keras.backend.clear_session()
        gc.collect()


class MLOnlyModelWrapper:
    """Wraps a sklearn/xgboost .joblib model that takes flattened patches."""

    def __init__(self, ml_path, name=""):
        self.name = name
        print(f"  Loading ML model: {name}")
        self.ml = joblib.load(ml_path)
        print(f"    ML classifier: {type(self.ml).__name__}")

    def predict_batch(self, patches):
        """patches: (N, H, W, C) → labels (N,)"""
        flat = patches.reshape(len(patches), -1)
        return self.ml.predict(flat)

    def cleanup(self):
        del self.ml
        gc.collect()


def load_model(model_config):
    """Load a model from its config tuple. Copies from Drive to /tmp first."""
    name, mtype, files = model_config
    local_files = [_copy_to_local(f"{MODELS_DIR}/{f}") for f in files]
    if mtype == 'keras':
        return KerasModelWrapper(local_files[0], name)
    elif mtype == 'hybrid':
        return HybridModelWrapper(local_files[0], local_files[1], name)
    elif mtype == 'ml':
        return MLOnlyModelWrapper(local_files[0], name)
    else:
        raise ValueError(f"Unknown model type: {mtype}")


print("✅ Model wrappers defined")

# %% ============================================================
# CELL 5: Sliding Window Classifier (Memory-Efficient)
# ============================================================
import rasterio
from numpy.lib.stride_tricks import as_strided


def classify_tile(model, raster_path, output_path, strip_rows=STRIP_ROWS):
    """
    Memory-efficient sliding-window classification of a GeoTIFF tile.

    Reads horizontal strips, extracts patches, classifies, writes directly
    to output. Peak RAM bounded by: strip_rows × width × n_bands × 4 bytes.
    """
    if Path(output_path).exists():
        print(f"    ⏭️  Already classified, skipping")
        return True

    with rasterio.open(raster_path) as src:
        h, w = src.height, src.width
        n_bands = src.count
        half = PATCH_SIZE // 2
        n_valid_rows = h - 2 * half
        n_valid_cols = w - 2 * half

        if n_valid_rows <= 0 or n_valid_cols <= 0:
            print(f"    ⚠️  Tile too small ({h}×{w}), skipping")
            return False

        # Create output file
        profile = src.profile.copy()
        profile.update(count=1, dtype='int8', nodata=-1,
                       compress='lzw', tiled=True,
                       blockxsize=256, blockysize=256)

        with rasterio.open(output_path, 'w', **profile) as dst:
            classified_full = np.full((h, w), -1, dtype=np.int8)

            processed = 0
            strip_start = 0
            t0 = time.time()

            while strip_start < n_valid_rows:
                strip_end = min(strip_start + strip_rows, n_valid_rows)

                # Read strip with overlap for patch extraction
                read_start = strip_start
                read_end = min(strip_end + PATCH_SIZE - 1, h)
                read_h = read_end - read_start

                window = rasterio.windows.Window(0, read_start, w, read_h)
                strip_data = src.read(window=window).astype(np.float32)

                # Normalize
                for b in range(n_bands):
                    strip_data[b] = (strip_data[b] - scaler_min[b]) / scaler_range[b]
                np.clip(strip_data, 0, 1, out=strip_data)

                strip_hwb = strip_data.transpose(1, 2, 0)  # (H, W, B)
                del strip_data

                n_patch_rows = strip_end - strip_start
                if strip_hwb.shape[0] < PATCH_SIZE:
                    strip_start = strip_end
                    del strip_hwb
                    continue

                # Use stride tricks for efficient patch extraction
                view_shape = (n_patch_rows, n_valid_cols,
                              PATCH_SIZE, PATCH_SIZE, n_bands)
                view_strides = (strip_hwb.strides[0], strip_hwb.strides[1],
                                strip_hwb.strides[0], strip_hwb.strides[1],
                                strip_hwb.strides[2])
                patches_view = as_strided(strip_hwb, shape=view_shape,
                                          strides=view_strides)

                # Process row by row within the strip
                for r_local in range(n_patch_rows):
                    r_global = strip_start + r_local
                    row_patches = np.ascontiguousarray(patches_view[r_local])

                    # Skip NaN pixels
                    nan_mask = np.any(
                        np.isnan(row_patches.reshape(n_valid_cols, -1)), axis=1)
                    valid_idx = np.where(~nan_mask)[0]
                    if len(valid_idx) == 0:
                        continue

                    labels = model.predict_batch(row_patches[valid_idx])
                    classified_full[r_global + half, valid_idx + half] = labels

                processed += n_patch_rows
                elapsed = time.time() - t0
                pct = processed / n_valid_rows * 100
                eta = (elapsed / processed) * (n_valid_rows - processed) if processed > 0 else 0
                print(f"\r    {processed}/{n_valid_rows} rows ({pct:.0f}%) "
                      f"ETA: {eta:.0f}s", end="", flush=True)

                del strip_hwb, patches_view
                gc.collect()
                strip_start = strip_end

            # Write full result
            dst.write(classified_full, 1)

        print(f"\n    ✅ Saved: {Path(output_path).name} "
              f"({time.time()-t0:.0f}s)")

    del classified_full
    gc.collect()
    return True


print("✅ Classifier function defined")

# %% ============================================================
# CELL 6: Run Classification for All Models × All Years
# ============================================================

# Choose which tiles to process:
#   'all'   — process every tile (slow but complete)
#   'large' — skip tiles < 100 MB (process only the main study area)
#   [0,1,5] — process specific tile indices only
TILE_SELECTION = 'all'

tile_dirs = {2018: STACK_2018_DIR, 2024: STACK_2024_DIR}

for model_cfg in MODELS_CONFIG:
    model_name = model_cfg[0]
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    # Load model once, process all years
    model = load_model(model_cfg)

    for year in YEARS:
        tile_dir = tile_dirs[year]
        if not os.path.exists(tile_dir):
            print(f"  ⚠️  {tile_dir} not found, skipping {year}")
            continue

        out_dir = Path(OUTPUT_DIR) / safe_name / f"classified_{year}"
        out_dir.mkdir(parents=True, exist_ok=True)

        tiles = sorted(Path(tile_dir).glob("*.tif"))
        print(f"\n  Year {year}: {len(tiles)} tiles in {tile_dir}")

        # Filter tiles based on selection
        if TILE_SELECTION == 'large':
            tiles = [t for t in tiles if t.stat().st_size > 100_000_000]
            print(f"  Filtered to {len(tiles)} tiles > 100 MB")
        elif isinstance(TILE_SELECTION, list):
            tiles = [tiles[i] for i in TILE_SELECTION if i < len(tiles)]
            print(f"  Processing {len(tiles)} selected tiles")

        for ti, tile_path in enumerate(tiles, 1):
            size_mb = tile_path.stat().st_size / (1024 * 1024)
            print(f"\n  [{ti}/{len(tiles)}] {tile_path.name} ({size_mb:.0f} MB)")

            out_path = out_dir / tile_path.name
            try:
                classify_tile(model, str(tile_path), str(out_path))
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue

    # Free model memory before loading next
    model.cleanup()
    gc.collect()

print(f"\n{'='*60}")
print("✅ ALL CLASSIFICATIONS COMPLETE")
print(f"{'='*60}")

# %% ============================================================
# CELL 7: Compute Area Statistics
# ============================================================

print("\nComputing area statistics per model per year...\n")

stats_all = {}

for model_cfg in MODELS_CONFIG:
    model_name = model_cfg[0]
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    stats_all[model_name] = {}

    for year in YEARS:
        cls_dir = Path(OUTPUT_DIR) / safe_name / f"classified_{year}"
        if not cls_dir.exists():
            continue

        year_stats = {name: {'area_km2': 0.0, 'pixel_count': 0}
                      for name in CLASS_NAMES}

        for tif in sorted(cls_dir.glob("*.tif")):
            with rasterio.open(tif) as src:
                data = src.read(1)
                transform = src.transform
                pixel_area_km2 = abs(transform.a * transform.e) / 1e6

            for i, name in enumerate(CLASS_NAMES):
                count = int(np.sum(data == i))
                year_stats[name]['pixel_count'] += count
                year_stats[name]['area_km2'] += count * pixel_area_km2

        total = sum(s['pixel_count'] for s in year_stats.values())
        if total > 0:
            for name in CLASS_NAMES:
                year_stats[name]['percentage'] = (
                    year_stats[name]['pixel_count'] / total * 100)

        stats_all[model_name][year] = year_stats

        # Save per model/year
        stats_path = cls_dir.parent / f"area_stats_{year}.json"
        with open(stats_path, 'w') as f:
            json.dump(year_stats, f, indent=2, default=str)

        print(f"  {model_name} — {year}:")
        for name, s in year_stats.items():
            print(f"    {name}: {s['area_km2']:.2f} km² ({s.get('percentage',0):.1f}%)")
        print()

print("✅ Area statistics saved")

# %% ============================================================
# CELL 8: Render Publication-Quality Maps (5 types per model)
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

cmap = ListedColormap(CLASS_COLORS[:N_CLASSES])
cmap.set_bad(color='black')
FOREST_CLASS = 0

rendered_dir = Path(OUTPUT_DIR) / "rendered_maps"
rendered_dir.mkdir(exist_ok=True)


def _add_cartographic_elements(ax, src_transform, src_h, src_w):
    """Add north arrow and scale bar to a map axis."""
    # --- North Arrow ---
    x_arrow = 0.95
    y_arrow = 0.92
    ax.annotate('N', xy=(x_arrow, y_arrow), xycoords='axes fraction',
                ha='center', va='bottom', fontsize=14, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                xytext=(x_arrow, y_arrow - 0.08))

    # --- Scale Bar ---
    pixel_size_m = abs(src_transform.a)  # meters per pixel
    bar_length_km = 5  # 5 km scale bar
    bar_length_px = int((bar_length_km * 1000) / pixel_size_m)
    fontprops = fm.FontProperties(size=10, weight='bold')
    scalebar = AnchoredSizeBar(
        ax.transData, bar_length_px, f'{bar_length_km} km',
        'lower left', pad=0.5, color='black', frameon=True,
        size_vertical=max(2, src_h // 300),
        fontproperties=fontprops,
        sep=5, fill_bar=True,
    )
    ax.add_artist(scalebar)


def render_model_maps(model_name, safe_name, year, map_type='lulc'):
    """
    Render a single publication-quality map for one model/year.
    map_type: 'lulc', 'forest', or 'deforestation'
    """
    cls_dir = Path(OUTPUT_DIR) / safe_name / f"classified_{year}"
    if not cls_dir.exists():
        return None

    # Use the largest tile as representative
    tiles = sorted(cls_dir.glob("*.tif"),
                   key=lambda t: t.stat().st_size, reverse=True)
    if not tiles:
        return None

    with rasterio.open(tiles[0]) as src:
        data = src.read(1)
        transform = src.transform
        h, w = src.height, src.width

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    if map_type == 'lulc':
        masked = np.ma.masked_where(data < 0, data)
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=N_CLASSES-1,
                  interpolation='nearest')
        ax.set_title(f'{model_name} — LULC Classification {year}',
                     fontsize=16, fontweight='bold')
        patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
                   for i in range(N_CLASSES)]
        ax.legend(handles=patches, loc='lower right', fontsize=11,
                  framealpha=0.95, title='Land Cover', title_fontsize=12)
        suffix = f"LULC_{year}"

    elif map_type == 'forest':
        forest = np.where(data == FOREST_CLASS, 1, 0).astype(np.float32)
        forest = np.ma.masked_where((data < 0) | (data != FOREST_CLASS), forest)
        # Light gray background for non-forest
        bg = np.ones_like(data, dtype=np.float32) * 0.85
        bg = np.ma.masked_where(data < 0, bg)
        ax.imshow(bg, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        forest_cmap = ListedColormap(['#00FF00'])
        ax.imshow(forest, cmap=forest_cmap, vmin=0.5, vmax=1.5,
                  interpolation='nearest', alpha=0.9)
        ax.set_title(f'{model_name} — Forest Cover {year}',
                     fontsize=16, fontweight='bold')
        patches = [mpatches.Patch(color='#00FF00', label='Forest'),
                   mpatches.Patch(color='#d9d9d9', label='Non-Forest')]
        ax.legend(handles=patches, loc='lower right', fontsize=11,
                  framealpha=0.95, title='Forest Cover', title_fontsize=12)
        suffix = f"Forest_{year}"

    elif map_type == 'deforestation':
        cls_2018_dir = Path(OUTPUT_DIR) / safe_name / "classified_2018"
        cls_2024_dir = Path(OUTPUT_DIR) / safe_name / "classified_2024"
        if not cls_2018_dir.exists() or not cls_2024_dir.exists():
            plt.close(fig)
            return None
        t18 = sorted(cls_2018_dir.glob("*.tif"),
                     key=lambda t: t.stat().st_size, reverse=True)
        t24 = sorted(cls_2024_dir.glob("*.tif"),
                     key=lambda t: t.stat().st_size, reverse=True)
        if not t18 or not t24:
            plt.close(fig)
            return None
        with rasterio.open(t18[0]) as s:
            c18 = s.read(1)
            transform = s.transform
            h, w = s.height, s.width
        with rasterio.open(t24[0]) as s:
            c24 = s.read(1)
        deforest = ((c18 == FOREST_CLASS) & (c24 != FOREST_CLASS))
        # Gray background
        bg = np.ones((h, w), dtype=np.float32) * 0.85
        bg = np.ma.masked_where(c18 < 0, bg)
        ax.imshow(bg, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        deforest_masked = np.ma.masked_where(~deforest, deforest.astype(float))
        ax.imshow(deforest_masked, cmap=ListedColormap(['#FFFF00']),
                  vmin=0.5, vmax=1.5, interpolation='nearest', alpha=0.9)
        ax.set_title(f'{model_name} — Deforestation 2018–2024',
                     fontsize=16, fontweight='bold')
        patches = [mpatches.Patch(color='#FFFF00', label='Deforested'),
                   mpatches.Patch(color='#d9d9d9', label='No Change')]
        ax.legend(handles=patches, loc='lower right', fontsize=11,
                  framealpha=0.95, title='Change Detection', title_fontsize=12)
        suffix = "Deforestation_2018_2024"

    ax.axis('off')
    _add_cartographic_elements(ax, transform, h, w)

    out = rendered_dir / f"{safe_name}_{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


# --- Generate all 5 map types per model ---
print("\n🗺️  Rendering publication maps (5 types × each model)...\n")

for model_cfg in MODELS_CONFIG:
    model_name = model_cfg[0]
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    print(f"\n  {model_name}:")

    for year in YEARS:
        render_model_maps(model_name, safe_name, year, 'lulc')
        render_model_maps(model_name, safe_name, year, 'forest')

    render_model_maps(model_name, safe_name, None, 'deforestation')

print(f"\n✅ All publication maps saved to: {rendered_dir}")

# %% ============================================================
# CELL 9: Deforestation GeoTIFFs + Forest-Only GeoTIFFs
# ============================================================

print("\nGenerating deforestation & forest-only GeoTIFFs...\n")

for model_cfg in MODELS_CONFIG:
    model_name = model_cfg[0]
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")

    cls_2018 = Path(OUTPUT_DIR) / safe_name / "classified_2018"
    cls_2024 = Path(OUTPUT_DIR) / safe_name / "classified_2024"
    def_dir = Path(OUTPUT_DIR) / safe_name / "deforestation"
    forest_dir = Path(OUTPUT_DIR) / safe_name / "forest_only"
    def_dir.mkdir(exist_ok=True)
    forest_dir.mkdir(exist_ok=True)

    # Forest-only GeoTIFFs per year
    for year in YEARS:
        cls_dir = Path(OUTPUT_DIR) / safe_name / f"classified_{year}"
        if not cls_dir.exists():
            continue
        for tif in sorted(cls_dir.glob("*.tif")):
            with rasterio.open(tif) as src:
                data = src.read(1)
                profile = src.profile.copy()
            forest_mask = np.where(data == FOREST_CLASS, 1, 0).astype(np.int8)
            forest_mask[data < 0] = -1
            profile.update(count=1, dtype='int8', nodata=-1)
            out_f = forest_dir / f"forest_{year}_{tif.name}"
            with rasterio.open(str(out_f), 'w', **profile) as dst:
                dst.write(forest_mask, 1)

    # Deforestation GeoTIFFs
    if not cls_2018.exists() or not cls_2024.exists():
        print(f"  ⚠️  {model_name}: skipping deforestation (missing year)")
        continue

    total_def_km2 = 0.0
    for tif_2018 in sorted(cls_2018.glob("*.tif")):
        tif_2024 = cls_2024 / tif_2018.name.replace("2018", "2024")
        if not tif_2024.exists():
            continue
        with rasterio.open(tif_2018) as src:
            c18 = src.read(1)
            transform = src.transform
            profile = src.profile.copy()
        with rasterio.open(tif_2024) as src:
            c24 = src.read(1)
        deforest = ((c18 == FOREST_CLASS) & (c24 != FOREST_CLASS)).astype(np.int8)
        profile.update(count=1, dtype='int8', nodata=-1)
        out_def = def_dir / f"deforest_{tif_2018.name}"
        with rasterio.open(str(out_def), 'w', **profile) as dst:
            dst.write(deforest, 1)
        pixel_area_km2 = abs(transform.a * transform.e) / 1e6
        total_def_km2 += int(np.sum(deforest == 1)) * pixel_area_km2

    print(f"  {model_name}: {total_def_km2:.2f} km² deforested")

print("\n✅ All GeoTIFFs ready for GEE upload")

# %% ============================================================
# CELL 10: Summary
# ============================================================
print("\n" + "="*70)
print("  CLASSIFICATION MAP GENERATION — SUMMARY")
print("="*70)

for model_cfg in MODELS_CONFIG:
    model_name = model_cfg[0]
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    print(f"\n  📊 {model_name}")
    for year in YEARS:
        cls_dir = Path(OUTPUT_DIR) / safe_name / f"classified_{year}"
        if cls_dir.exists():
            n = len(list(cls_dir.glob("*.tif")))
            print(f"     {year}: {n} tiles classified")

print(f"\n  📁 Outputs: {OUTPUT_DIR}")
print(f"  🗺️  Maps:    {rendered_dir}")
print(f"\n  5 maps per model: LULC 2018, LULC 2024, Forest 2018, Forest 2024, Deforestation")
print(f"  GeoTIFFs ready for GEE upload (forest_only/ and deforestation/ folders)")
print("="*70)
