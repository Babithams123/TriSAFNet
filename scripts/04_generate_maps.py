"""
04_generate_maps.py
Train TriSAFNet on all combined data (single pass, no CV) and apply to
full rasters to produce classification, confidence, and deforestation maps.

Usage:
    python scripts/04_generate_maps.py

Outputs:
    outputs/classified_2018/   (per-tile classified GeoTIFFs)
    outputs/classified_2024/
    outputs/confidence_2018/   (per-tile confidence GeoTIFFs)
    outputs/confidence_2024/
    outputs/deforestation_2018_2024/
    outputs/deforestation_confidence.tif
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from numpy.lib.stride_tricks import as_strided
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import *
from models import TriSAFNet
import configs.config as _cfg_module
from utils.mlflow_utils import (
    init_mlflow, start_run, end_run, log_config, log_metrics,
    log_artifact_safe, log_artifacts_dir,
    build_artifact_manifest,
)


class SavedTriSAFNet:
    """Thin wrapper around a saved Keras model providing predict + confidence."""

    def __init__(self, model_path):
        print(f"  Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print(f"  Model loaded. Parameters: {self.model.count_params():,}")

    def predict(self, X):
        return np.argmax(self.model.predict(X, batch_size=256, verbose=0), axis=1)

    def predict_with_confidence(self, X):
        proba = self.model.predict(X, batch_size=512, verbose=0)
        labels = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)
        return labels, confidence


def sliding_window_predict_windowed(model, src, scaler_min, scaler_max,
                                    patch_size=PATCH_SIZE, batch_size=1024,
                                    strip_rows=256):
    """
    Memory-efficient sliding-window classification using rasterio windowed reads.

    Instead of loading the entire tile into RAM, reads horizontal strips of
    *strip_rows* scanlines (plus patch_size overlap) and classifies each strip
    independently.  Peak RAM per tile is bounded to roughly:
        strip_rows * width * n_bands * 4 bytes  (~300-500 MB for typical tiles)
    regardless of total tile height.
    """
    n_bands = src.count
    h = src.height
    w = src.width
    half = patch_size // 2
    ps = patch_size

    classified = np.full((h, w), -1, dtype=np.int8)
    confidence_map = np.full((h, w), 0.0, dtype=np.float32)

    ranges = (scaler_max - scaler_min).astype(np.float32)
    ranges[ranges == 0] = 1
    scaler_min_f = scaler_min.astype(np.float32)

    total_valid_rows = h - 2 * half
    n_cols = w - 2 * half
    if total_valid_rows <= 0 or n_cols <= 0:
        print(f"  Tile too small ({h}x{w}) for patch_size={ps}, skipping.")
        return classified, confidence_map

    print(f"  Classifying {h}x{w} raster ({total_valid_rows} scanlines, "
          f"{n_cols} valid cols) in strips of {strip_rows} rows...")

    processed_rows = 0
    strip_start = 0

    while strip_start < total_valid_rows:
        strip_end = min(strip_start + strip_rows, total_valid_rows)

        read_row_start = strip_start
        read_row_end = strip_end + ps - 1
        read_row_end = min(read_row_end, h)
        read_height = read_row_end - read_row_start

        window = rasterio.windows.Window(0, read_row_start, w, read_height)
        strip_data = src.read(window=window).astype(np.float32)

        for b in range(n_bands):
            strip_data[b] = (strip_data[b] - scaler_min_f[b]) / ranges[b]
        np.clip(strip_data, 0, 1, out=strip_data)

        strip_hwb = strip_data.transpose(1, 2, 0)
        del strip_data

        strip_h = strip_hwb.shape[0]
        n_patch_rows = strip_end - strip_start
        if strip_h < ps or n_cols <= 0:
            strip_start = strip_end
            del strip_hwb
            continue

        view_shape = (n_patch_rows, n_cols, ps, ps, n_bands)
        view_strides = (strip_hwb.strides[0], strip_hwb.strides[1],
                        strip_hwb.strides[0], strip_hwb.strides[1],
                        strip_hwb.strides[2])
        patches_view = as_strided(strip_hwb, shape=view_shape, strides=view_strides)

        for r_local in range(n_patch_rows):
            r_global = strip_start + r_local
            row = np.ascontiguousarray(patches_view[r_local])

            nan_mask = np.any(np.isnan(row.reshape(n_cols, -1)), axis=1)
            valid = np.where(~nan_mask)[0]
            if len(valid) == 0:
                continue

            labels, confs = model.predict_with_confidence(row[valid])
            classified[r_global + half, valid + half] = labels
            confidence_map[r_global + half, valid + half] = confs

        processed_rows += n_patch_rows
        if processed_rows % max(1, (total_valid_rows // 20)) < strip_rows or strip_end == total_valid_rows:
            print(f"    Rows {processed_rows}/{total_valid_rows} "
                  f"({processed_rows / total_valid_rows * 100:.1f}%)")

        del strip_hwb, patches_view
        gc.collect()

        strip_start = strip_end

    return classified, confidence_map


def save_classified_raster(data, reference_path, output_path, dtype='int8', nodata=-1):
    """Save a single-band raster using a reference for CRS/transform."""
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()

    profile.update(count=1, dtype=dtype, nodata=nodata)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(dtype), 1)
    print(f"  Saved: {output_path}")


def save_float_raster(data, reference_path, output_path):
    """Save a float32 raster (confidence map)."""
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype='float32', nodata=0.0)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(np.float32), 1)
    print(f"  Saved: {output_path}")


def class_name_from_id(class_id):
    class_id = int(class_id)
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"Class {class_id}"


def compute_deforestation(classified_2018, classified_2024, forest_class_id=0):
    """Deforestation = Forest in 2018 AND not Forest in 2024."""
    return ((classified_2018 == forest_class_id) & (classified_2024 != forest_class_id)).astype(np.int8)


def compute_area_statistics(classified, transform, class_names=CLASS_NAMES):
    """Compute area (km2) for each class."""
    pixel_area_km2 = abs(transform.a * transform.e) / 1e6
    stats = {}
    total_valid = np.sum(classified >= 0)
    for i, name in enumerate(class_names):
        count = int(np.sum(classified == i))
        area_km2 = count * pixel_area_km2
        pct = count / total_valid * 100 if total_valid > 0 else 0
        stats[name] = {'area_km2': area_km2, 'percentage': pct, 'pixel_count': count}
    return stats


def process_tile(model, raster_path, out_cls_path, out_conf_path,
                 scaler_min, scaler_max, patch_size):
    """Classify a single tile using memory-efficient windowed reads."""
    with rasterio.open(raster_path) as src:
        if src.count != len(scaler_min):
            raise ValueError(f"Band mismatch: tile has {src.count} bands, "
                             f"but scaler expects {len(scaler_min)} bands")
        transform = src.transform
        classified, confidence = sliding_window_predict_windowed(
            model, src, scaler_min, scaler_max, patch_size
        )
    save_classified_raster(classified, str(raster_path), str(out_cls_path))
    save_float_raster(confidence, str(raster_path), str(out_conf_path))
    gc.collect()
    return classified, confidence, transform


def render_classification_map(classified, year, output_path,
                              class_names=CLASS_NAMES, class_colors=CLASS_COLORS):
    """Render a classified raster tile as a colour-coded PNG."""
    masked = np.ma.masked_where(classified < 0, classified)
    cmap = ListedColormap(class_colors[:len(class_names)])
    cmap.set_bad(color='black')

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=len(class_names) - 1, interpolation='nearest')
    ax.set_title(f'LULC Classification — {year}', fontsize=14, fontweight='bold')
    ax.axis('off')

    patches = [mpatches.Patch(color=class_colors[i], label=n) for i, n in enumerate(class_names)]
    ax.legend(handles=patches, loc='lower right', fontsize=9, framealpha=0.9)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved rendered map: {output_path}")


def render_confidence_map(confidence, year, output_path):
    """Render a confidence map as a viridis-colored PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    masked = np.ma.masked_where(confidence == 0, confidence)
    im = ax.imshow(masked, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(f'Prediction Confidence — {year}', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Confidence')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved rendered map: {output_path}")


def render_deforestation_map(deforest, output_path):
    """Render a binary deforestation map as a red/green PNG."""
    cmap = ListedColormap(['#228B22', '#FF0000'])
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(deforest, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('Deforestation 2018–2024', fontsize=14, fontweight='bold')
    ax.axis('off')
    patches = [mpatches.Patch(color='#228B22', label='No Change'),
               mpatches.Patch(color='#FF0000', label='Deforested')]
    ax.legend(handles=patches, loc='lower right', fontsize=9, framealpha=0.9)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved rendered map: {output_path}")


def _safe_load_object_array(data, key, fallback):
    """Load a potentially-pickled NPZ field, returning *fallback* on error."""
    if key not in data.files:
        return fallback
    try:
        return data[key].tolist()
    except (ModuleNotFoundError, ImportError):
        return fallback


def load_npz_with_meta(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    patches = data['patches']
    labels = data['labels']
    n_bands = int(patches.shape[-1])
    meta = {
        'n_bands': n_bands,
        'patch_size': int(data['patch_size']) if 'patch_size' in data.files else int(patches.shape[1]),
        'class_ids': _safe_load_object_array(
            data, 'class_ids',
            [int(c) for c in np.unique(labels)]
        ),
        'band_names': _safe_load_object_array(
            data, 'band_names',
            BAND_NAMES[:n_bands] if n_bands <= len(BAND_NAMES) else [f"band_{i+1:02d}" for i in range(n_bands)]
        ),
    }
    return data, meta


def main(patch_size):
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("GENERATING CLASSIFICATION, CONFIDENCE & DEFORESTATION MAPS")
    print("=" * 60)

    # ---- TRAIN TriSAFNet ON ALL COMBINED DATA (single pass) ----
    print("\nLoading training data from both years...")

    X_parts, y_parts = [], []
    scaler_min, scaler_max = None, None

    metas = {}
    for year in [2018, 2024]:
        npz_path = f"{DATA_DIR}/patches_{year}_{patch_size}x{patch_size}.npz"
        if not Path(npz_path).exists():
            print(f"  WARNING: {npz_path} not found, skipping.")
            continue
        data, meta = load_npz_with_meta(npz_path)
        metas[year] = meta
        X_parts.append(np.array(data['patches']))
        y_parts.append(np.array(data['labels']))
        if scaler_min is None:
            scaler_min = np.array(data['scaler_min'])
            scaler_max = np.array(data['scaler_max'])
            print(f"  Using schema from {year}: bands={meta['n_bands']}, patch={meta['patch_size']}, "
                  f"classes={meta['class_ids']}")
        elif len(data['scaler_min']) != len(scaler_min):
            print(f"  WARNING: {year} scaler band count differs; keeping first scaler.")
        n_loaded = X_parts[-1].shape[0]
        data.close()
        print(f"  Loaded {year}: {n_loaded} patches")

    if not X_parts:
        raise FileNotFoundError(
            f"No patch datasets found for patch size {patch_size} in {DATA_DIR} "
            "(expected patches_2018_* and/or patches_2024_*)"
        )

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    del X_parts, y_parts
    gc.collect()

    input_shape = X_all.shape[1:]
    print(f"  Combined dataset: {X_all.shape[0]} patches, shape={input_shape}")

    model_dir = Path(OUTPUT_DIR) / "trisafnet_full_model"
    model_dir.mkdir(exist_ok=True, parents=True)
    final_model_path = model_dir / "best_model_full.keras"

    if final_model_path.exists():
        print(f"\n  Existing trained model found: {final_model_path}")
        print("  Skipping retraining. Delete the file to retrain.")
    else:
        print(f"\n  Training TriSAFNet on ALL {X_all.shape[0]} samples (single pass)...")
        tf.keras.backend.clear_session()
        model_instance = TriSAFNet(input_shape)
        model_instance.fit(X_all, y_all)
        model_instance.model.save(str(final_model_path))
        print(f"  Model saved: {final_model_path}")

        meta = {
            'n_samples': int(X_all.shape[0]),
            'input_shape': list(map(int, input_shape)),
            'n_classes': len(np.unique(y_all)),
            'patch_size': int(patch_size),
            'n_bands': int(input_shape[-1]),
            'parameters': int(model_instance.get_param_count()),
            'class_ids': [int(c) for c in np.unique(y_all)],
            'band_names': list(next(iter(metas.values())).get('band_names', [])) if metas else [],
        }
        with open(model_dir / 'training_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

    # Runtime classes from merged data
    active_class_ids = [int(c) for c in np.unique(np.concatenate([np.array(m.get('class_ids', []), dtype=int) for m in metas.values()]))] if metas else [int(c) for c in np.unique(y_all)]
    active_class_names = [class_name_from_id(c) for c in active_class_ids]
    forest_class_id = active_class_ids[active_class_names.index("Forest")] if "Forest" in active_class_names else 0

    del X_all, y_all
    tf.keras.backend.clear_session()
    gc.collect()

    print("\nLoading best model for map generation...")
    model = SavedTriSAFNet(str(final_model_path))

    # ---- CLASSIFY + CONFIDENCE FOR EACH YEAR, TILE-BY-TILE ----
    for year in [2018, 2024]:
        tile_dir = Path(f"{DATA_DIR}/Stack_{year}")
        if not tile_dir.exists():
            print(f"Directory {tile_dir} not found. Skipping {year}.")
            continue

        out_cls_dir = output_dir / f"classified_{year}"
        out_cls_dir.mkdir(exist_ok=True)
        out_conf_dir = output_dir / f"confidence_{year}"
        out_conf_dir.mkdir(exist_ok=True)

        print(f"\nClassifying {year} tiles from: {tile_dir}")
        stats_all = {name: {'area_km2': 0.0, 'pixel_count': 0} for name in active_class_names}

        tiles = sorted(tile_dir.glob("*.tif"))
        print(f"  Found {len(tiles)} tiles")

        for ti, raster_path in enumerate(tiles, 1):
            out_cls = out_cls_dir / raster_path.name
            out_conf = out_conf_dir / raster_path.name

            if out_cls.exists() and out_conf.exists():
                print(f"  [{ti}/{len(tiles)}] Skipping {raster_path.name} (already done)")
                with rasterio.open(out_cls) as src:
                    classified = src.read(1)
                    transform = src.transform
            else:
                print(f"  [{ti}/{len(tiles)}] Processing tile: {raster_path.name}")
                try:
                    classified, confidence, transform = process_tile(
                        model, raster_path, out_cls, out_conf,
                        scaler_min, scaler_max, patch_size
                    )
                except ValueError as e:
                    print(f"  [{ti}/{len(tiles)}] Skipping {raster_path.name}: {e}")
                    continue

            stats = compute_area_statistics(classified, transform, class_names=active_class_names)
            for name in active_class_names:
                stats_all[name]['area_km2'] += stats[name]['area_km2']
                stats_all[name]['pixel_count'] += stats[name]['pixel_count']
            del classified
            gc.collect()

        total_pixels = sum(s['pixel_count'] for s in stats_all.values())
        if total_pixels > 0:
            for name in active_class_names:
                stats_all[name]['percentage'] = (stats_all[name]['pixel_count'] / total_pixels) * 100

        print(f"\n  Final Area Statistics for {year}:")
        for name, s in stats_all.items():
            print(f"    {name}: {s['area_km2']:.2f} km2 ({s.get('percentage', 0):.1f}%)")

        with open(f"{output_dir}/area_stats_{year}.json", 'w') as f:
            json.dump(stats_all, f, indent=2, default=str)

    # ---- DEFORESTATION MAP (parallel tile processing) ----
    print("\nComputing deforestation map per tile...")
    def_dir = output_dir / "deforestation_2018_2024"
    def_dir.mkdir(exist_ok=True)

    cls_2018_dir = output_dir / "classified_2018"
    cls_2024_dir = output_dir / "classified_2024"
    conf_2018_dir = output_dir / "confidence_2018"
    conf_2024_dir = output_dir / "confidence_2024"

    def _process_deforest_tile(c2018_path):
        c2024_name = c2018_path.name.replace("2018", "2024")
        c2024_path = cls_2024_dir / c2024_name
        if not c2024_path.exists():
            return 0.0

        with rasterio.open(c2018_path) as src:
            c2018 = src.read(1)
            transform = src.transform
        with rasterio.open(c2024_path) as src:
            c2024 = src.read(1)

        deforest = compute_deforestation(c2018, c2024, forest_class_id=forest_class_id)
        out_def = def_dir / f"deforest_{c2018_path.name}"
        save_classified_raster(deforest, str(c2018_path), str(out_def))

        conf18_path = conf_2018_dir / c2018_path.name
        conf24_path = conf_2024_dir / c2024_name
        if conf18_path.exists() and conf24_path.exists():
            with rasterio.open(conf18_path) as src:
                cf18 = src.read(1)
            with rasterio.open(conf24_path) as src:
                cf24 = src.read(1)
            def_conf = ((cf18 + cf24) / 2.0) * deforest.astype(np.float32)
            save_float_raster(def_conf, str(c2018_path),
                              str(def_dir / f"deforest_confidence_{c2018_path.name}"))

        return np.sum(deforest == 1) * abs(transform.a * transform.e) / 1e6

    tile_paths_2018 = sorted(cls_2018_dir.glob("*.tif")) if cls_2018_dir.exists() else []
    n_def_workers = min(4, max(1, os.cpu_count() or 1))
    total_def_area = 0.0
    with ThreadPoolExecutor(max_workers=n_def_workers) as pool:
        futs = [pool.submit(_process_deforest_tile, p) for p in tile_paths_2018]
        for fut in as_completed(futs):
            total_def_area += fut.result()
    print(f"  Total deforestation area: {total_def_area:.2f} km2")

    # ---- RENDER VISUAL MAP IMAGES (parallel) ----
    rendered_dir = output_dir / "rendered_maps"
    rendered_dir.mkdir(exist_ok=True)

    def _render_cls(year):
        cls_dir = output_dir / f"classified_{year}"
        if not cls_dir.exists():
            return
        first_cls = next(cls_dir.glob("*.tif"), None)
        if first_cls:
            with rasterio.open(first_cls) as src:
                cls_data = src.read(1)
            render_classification_map(
                cls_data, year, str(rendered_dir / f"classification_{year}.png"),
                class_names=active_class_names, class_colors=CLASS_COLORS[:len(active_class_names)],
            )

    def _render_conf(year):
        conf_dir = output_dir / f"confidence_{year}"
        if not conf_dir.exists():
            return
        first_conf = next(conf_dir.glob("*.tif"), None)
        if first_conf:
            with rasterio.open(first_conf) as src:
                conf_data = src.read(1)
            render_confidence_map(conf_data, year, str(rendered_dir / f"confidence_{year}.png"))

    def _render_deforest():
        first_def = next(def_dir.glob("deforest_*.tif"), None) if def_dir.exists() else None
        if first_def:
            with rasterio.open(first_def) as src:
                def_data = src.read(1)
            render_deforestation_map(def_data, str(rendered_dir / "deforestation_2018_2024.png"))

    render_jobs = [
        lambda: _render_cls(2018), lambda: _render_cls(2024),
        lambda: _render_conf(2018), lambda: _render_conf(2024),
        _render_deforest,
    ]
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(fn) for fn in render_jobs]
        for fut in as_completed(futs):
            fut.result()

    # ---- ARTIFACT MANIFEST ----
    expected_artifacts = {}
    for year in [2018, 2024]:
        expected_artifacts[f'area_stats_{year}'] = str(output_dir / f'area_stats_{year}.json')
        expected_artifacts[f'classification_map_{year}'] = str(rendered_dir / f'classification_{year}.png')
        expected_artifacts[f'confidence_map_{year}'] = str(rendered_dir / f'confidence_{year}.png')
    expected_artifacts['deforestation_map'] = str(rendered_dir / 'deforestation_2018_2024.png')
    expected_artifacts['trained_model'] = str(final_model_path)
    expected_artifacts['training_meta'] = str(model_dir / 'training_meta.json')
    manifest_path = build_artifact_manifest(str(output_dir), expected_artifacts)

    # ---- FINAL CLEANUP ----
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    print("\n" + "=" * 60)
    print("MAP GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  {output_dir}/classified_2018/       (classification GeoTIFFs)")
    print(f"  {output_dir}/classified_2024/")
    print(f"  {output_dir}/confidence_2018/       (confidence GeoTIFFs)")
    print(f"  {output_dir}/confidence_2024/")
    print(f"  {output_dir}/deforestation_2018_2024/")
    print(f"  {output_dir}/rendered_maps/         (visual PNG maps)")
    print(f"  {output_dir}/area_stats_*.json")
    print(f"  {output_dir}/artifact_manifest.json")

    return {
        'total_def_area_km2': total_def_area,
        'manifest_path': str(manifest_path),
        'output_dir': str(output_dir),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate LULC and deforestation maps")
    parser.add_argument('--patch-size', type=int, default=PATCH_SIZE,
                        help=f'Patch size used in NPZ/model training (default: {PATCH_SIZE})')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow tracking')
    args = parser.parse_args()

    # ---- MLFLOW INIT ----
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        try:
            init_mlflow(experiment_name="LULC_MapGeneration")
        except Exception as e:
            print(f"  MLflow init failed ({e}); continuing without MLflow.")
            use_mlflow = False

    _mlflow_run = None
    if use_mlflow:
        try:
            _mlflow_run = start_run(
                run_name=f"maps_ps{args.patch_size}",
                tags={'task': 'map_generation', 'patch_size': str(args.patch_size)},
            )
            log_config(_cfg_module)
        except Exception as e:
            print(f"  MLflow run start failed: {e}")

    try:
        result = main(args.patch_size)

        if use_mlflow and result:
            try:
                log_metrics({'deforestation_area_km2': result['total_def_area_km2']})

                out = Path(result['output_dir'])
                for stats_file in out.glob('area_stats_*.json'):
                    with open(stats_file) as _f:
                        stats = json.load(_f)
                    year_tag = stats_file.stem.split('_')[-1]
                    for cls_name, vals in stats.items():
                        safe = cls_name.replace(' ', '_')
                        log_metrics({
                            f'{year_tag}_{safe}_area_km2': vals.get('area_km2', 0),
                            f'{year_tag}_{safe}_pct': vals.get('percentage', 0),
                        })

                log_artifacts_dir(str(out / 'rendered_maps'), artifact_path='rendered_maps')
                for stats_file in out.glob('area_stats_*.json'):
                    log_artifact_safe(str(stats_file))
                log_artifact_safe(result['manifest_path'])
                log_artifact_safe(str(out / 'trisafnet_full_model' / 'training_meta.json'))

                print("\n  MLflow: all map-generation artifacts logged.")
            except Exception as e:
                print(f"  MLflow artifact logging failed: {e}")
    finally:
        if _mlflow_run is not None:
            try:
                end_run()
            except Exception:
                pass

    print("\nMLflow UI: run 'mlflow ui' to browse experiments.")
