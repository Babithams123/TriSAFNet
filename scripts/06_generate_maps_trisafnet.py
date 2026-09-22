"""
06_generate_maps_trisafnet.py
Load the best saved TriSAFNet model and generate classification &
deforestation maps from all TIFF tiles.

Usage:
    python scripts/06_generate_maps_trisafnet.py --year both
    python scripts/06_generate_maps_trisafnet.py --year 2018
    python scripts/06_generate_maps_trisafnet.py --year 2024
    python scripts/06_generate_maps_trisafnet.py --year both --model-path outputs/trisafnet_models/best_model_2018.keras

Outputs:
    outputs/classified_2018/   (per-tile classified GeoTIFFs)
    outputs/classified_2024/
    outputs/deforestation_2018_2024/
    outputs/area_stats_2018.json
    outputs/area_stats_2024.json
"""

import os, sys
_dll_dir = os.path.join(sys.prefix, 'Library', 'bin')
if os.path.isdir(_dll_dir):
    os.environ['PATH'] = _dll_dir + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(_dll_dir)
_cuda_data_dir = os.path.join(sys.prefix, 'Library')
if os.path.isfile(os.path.join(_cuda_data_dir, 'bin', 'libdevice.10.bc')):
    os.environ.setdefault('XLA_FLAGS', f'--xla_gpu_cuda_data_dir={_cuda_data_dir}')

import argparse
import gc
import json
import numpy as np
import rasterio
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import *

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# ============================================================
# MODEL WRAPPER
# ============================================================

class SavedTriSAFNet:
    """Thin wrapper around a saved Keras model to provide the same
    predict interface used by the sliding window function."""

    def __init__(self, model_path):
        print(f"  Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print(f"  Model loaded. Parameters: {self.model.count_params():,}")

    def predict(self, X):
        return np.argmax(self.model.predict(X, batch_size=256, verbose=0), axis=1)


# ============================================================
# SLIDING WINDOW INFERENCE
# ============================================================

def sliding_window_predict(model, raster_data, scaler_min, scaler_max,
                           patch_size=PATCH_SIZE, batch_size=512):
    """
    Apply model to full raster using sliding window inference.

    Args:
        model: object with .predict(X) -> class labels
        raster_data: (bands, height, width) array
        scaler_min: 1st percentile array from training
        scaler_max: 99th percentile array from training
        patch_size: window size
        batch_size: prediction batch size

    Returns:
        classified: (height, width) array of class labels
    """
    n_bands, h, w = raster_data.shape
    half = patch_size // 2
    classified = np.full((h, w), -1, dtype=np.int8)

    # Robust normalization using training data stats
    ranges = scaler_max - scaler_min
    ranges[ranges == 0] = 1

    raster_norm = np.zeros_like(raster_data, dtype=np.float32)
    for b in range(n_bands):
        raster_norm[b] = (raster_data[b] - scaler_min[b]) / ranges[b]

    raster_norm = np.clip(raster_norm, 0, 1)

    # Collect patches
    print(f"  Generating patches for {h}x{w} raster...")
    rows = range(half, h - half, 1)
    cols = range(half, w - half, 1)

    total_pixels = len(rows) * len(cols)
    print(f"  Total pixels to classify: {total_pixels}")

    batch_patches = []
    batch_coords = []
    n_processed = 0

    for r in rows:
        for c in cols:
            patch = raster_norm[:, r-half:r+half+1, c-half:c+half+1]

            if np.any(np.isnan(patch)):
                continue

            batch_patches.append(patch.transpose(1, 2, 0))
            batch_coords.append((r, c))

            if len(batch_patches) >= batch_size:
                X_batch = np.array(batch_patches, dtype=np.float32)
                preds = model.predict(X_batch)
                for (pr, pc), pred in zip(batch_coords, preds):
                    classified[pr, pc] = pred

                n_processed += len(batch_patches)
                if n_processed % 50000 == 0:
                    pct = n_processed / total_pixels * 100
                    print(f"    Progress: {n_processed}/{total_pixels} ({pct:.1f}%)")

                batch_patches = []
                batch_coords = []

    # Process remaining
    if batch_patches:
        X_batch = np.array(batch_patches, dtype=np.float32)
        preds = model.predict(X_batch)
        for (pr, pc), pred in zip(batch_coords, preds):
            classified[pr, pc] = pred

    return classified


# ============================================================
# RASTER I/O
# ============================================================

def save_classified_raster(classified, reference_path, output_path):
    """Save classification result as GeoTIFF."""
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()

    profile.update(
        count=1,
        dtype='int8',
        nodata=-1
    )

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(classified.astype(np.int8), 1)

    print(f"  Saved: {output_path}")


def compute_deforestation(classified_2018, classified_2024):
    """
    Compute deforestation map.
    Deforestation = Forest in 2018 (class 0) AND NOT Forest in 2024.
    """
    forest_2018 = (classified_2018 == 0)
    forest_2024 = (classified_2024 == 0)
    deforestation = (forest_2018 & ~forest_2024).astype(np.int8)
    return deforestation


def compute_area_statistics(classified, transform, class_names=CLASS_NAMES):
    """Compute area (km²) for each class."""
    pixel_area_m2 = abs(transform.a * transform.e)
    pixel_area_km2 = pixel_area_m2 / 1e6

    stats = {}
    total_valid = np.sum(classified >= 0)

    for i, name in enumerate(class_names):
        count = np.sum(classified == i)
        area_km2 = count * pixel_area_km2
        pct = count / total_valid * 100 if total_valid > 0 else 0
        stats[name] = {'area_km2': area_km2, 'percentage': pct, 'pixel_count': int(count)}

    return stats


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate classification maps using saved TriSAFNet model')
    parser.add_argument('--year', type=str, default='both',
                        choices=['2018', '2024', 'both'])
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved .keras model. '
                             'Defaults to outputs/trisafnet_models/best_model_{year}.keras')
    args = parser.parse_args()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    years = [2018, 2024] if args.year == 'both' else [int(args.year)]

    print("=" * 60)
    print("GENERATING CLASSIFICATION & DEFORESTATION MAPS")
    print("  Using saved TriSAFNet model")
    print("=" * 60)

    # ---- LOAD SCALER STATS FROM TRAINING DATA ----
    print("\nLoading scaler statistics from training data...")
    scaler_min, scaler_max = None, None
    for year in [2018, 2024]:
        npz_path = f"{DATA_DIR}/patches_{year}_{PATCH_SIZE}x{PATCH_SIZE}.npz"
        if Path(npz_path).exists():
            data = np.load(npz_path)
            if scaler_min is None:
                scaler_min = data['scaler_min']
                scaler_max = data['scaler_max']
            print(f"  Loaded scaler from {year} data")
            break

    if scaler_min is None:
        print(f"ERROR: No patches_*_{PATCH_SIZE}x{PATCH_SIZE}.npz found in data/.")
        sys.exit(1)

    # ---- CLASSIFY TILES FOR EACH YEAR ----
    for year in years:
        # Determine model path
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = str(Path(OUTPUT_DIR) / "trisafnet_models" / f"best_model_{year}.keras")

        if not Path(model_path).exists():
            print(f"\nERROR: Model not found at {model_path}")
            print("  Run 05_train_trisafnet.py first, or specify --model-path")
            continue

        # Load model
        print(f"\n--- Loading TriSAFNet for {year} ---")
        model = SavedTriSAFNet(model_path)

        # Load manifest if available
        manifest_path = Path(model_path).parent / f"best_model_{year}.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            print(f"  Best fold: {manifest.get('best_fold', '?')}, "
                  f"Accuracy: {manifest.get('accuracy', '?'):.4f}, "
                  f"F1: {manifest.get('f1_macro', '?'):.4f}")

        # Find tiles
        tile_dir = Path(f"{DATA_DIR}/Stack_{year}")
        if not tile_dir.exists():
            print(f"  Directory {tile_dir} not found. Skipping {year}.")
            continue

        out_year_dir = output_dir / f"classified_{year}"
        out_year_dir.mkdir(exist_ok=True)

        print(f"\nClassifying {year} tiles from: {tile_dir}")
        stats_all = {name: {'area_km2': 0.0, 'pixel_count': 0, 'percentage': 0.0}
                     for name in CLASS_NAMES}

        tiles = sorted(tile_dir.glob("*.tif"))
        print(f"  Found {len(tiles)} tiles")

        for ti, raster_path in enumerate(tiles, 1):
            out_path = out_year_dir / raster_path.name
            if out_path.exists():
                print(f"  [{ti}/{len(tiles)}] Skipping {raster_path.name} (already done)")
                with rasterio.open(out_path) as src:
                    classified = src.read(1)
                    transform = src.transform
            else:
                print(f"  [{ti}/{len(tiles)}] Processing tile: {raster_path.name}")
                with rasterio.open(raster_path) as src:
                    raster_data = src.read().astype(np.float32)
                    transform = src.transform

                classified = sliding_window_predict(
                    model, raster_data, scaler_min, scaler_max, PATCH_SIZE)
                save_classified_raster(classified, str(raster_path), str(out_path))
                del raster_data
                gc.collect()

            # Aggregate area statistics
            stats = compute_area_statistics(classified, transform)
            for name in CLASS_NAMES:
                stats_all[name]['area_km2'] += stats[name]['area_km2']
                stats_all[name]['pixel_count'] += stats[name]['pixel_count']

        # Compute final percentages
        total_pixels = sum(s['pixel_count'] for s in stats_all.values())
        if total_pixels > 0:
            for name in CLASS_NAMES:
                stats_all[name]['percentage'] = (
                    stats_all[name]['pixel_count'] / total_pixels) * 100

        print(f"\n  Final Area Statistics for {year}:")
        for name, s in stats_all.items():
            print(f"    {name}: {s['area_km2']:.2f} km² ({s['percentage']:.1f}%)")

        with open(f"{output_dir}/area_stats_{year}.json", 'w') as f:
            json.dump(stats_all, f, indent=2, default=str)

        # Clean up model to free GPU memory before next year
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # ---- DEFORESTATION MAP ----
    if len(years) == 2 or (
        (output_dir / "classified_2018").exists() and
        (output_dir / "classified_2024").exists()
    ):
        print("\nComputing deforestation map per tile...")
        def_dir = output_dir / "deforestation_2018_2024"
        def_dir.mkdir(exist_ok=True)

        total_def_area = 0

        for c2018_path in (output_dir / "classified_2018").glob("*.tif"):
            c2024_name = c2018_path.name.replace("2018", "2024")
            c2024_path = output_dir / "classified_2024" / c2024_name

            if not c2024_path.exists():
                continue

            with rasterio.open(c2018_path) as src:
                c2018 = src.read(1)
                transform = src.transform
            with rasterio.open(c2024_path) as src:
                c2024 = src.read(1)

            deforest = compute_deforestation(c2018, c2024)
            out_def_path = def_dir / f"deforest_{c2018_path.name}"
            save_classified_raster(deforest, str(c2018_path), str(out_def_path))

            total_def_area += np.sum(deforest == 1) * abs(
                transform.a * transform.e) / 1e6

        print(f"  Total deforestation area: {total_def_area:.2f} km²")

    print("\n" + "=" * 60)
    print("MAP GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs written to:")
    print(f"  {output_dir}/classified_2018/")
    print(f"  {output_dir}/classified_2024/")
    print(f"  {output_dir}/deforestation_2018_2024/")
    print(f"  {output_dir}/area_stats_2018.json")
    print(f"  {output_dir}/area_stats_2024.json")
    print(f"\nOpen these in QGIS to create publication-quality maps with:")
    print(f"  - Legend, scale bar, north arrow")
    print(f"  - Proper color scheme matching CLASS_COLORS in config.py")
    print(f"  - Karnataka boundary overlay")


if __name__ == '__main__':
    main()
