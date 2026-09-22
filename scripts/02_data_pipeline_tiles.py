"""
02_data_pipeline_tiles.py
Load GeoTIFF tiles directly from data/Stack_2018 and data/Stack_2024
(without needing a merged raster), extract patches around labeled
training points from each tile, and prepare train/test splits with
spatial separation to prevent data leakage.

This avoids the memory-intensive merge step by processing each tile
independently -- only points falling within a tile's extent are used.

Usage:
    python scripts/02_data_pipeline_tiles.py

    # Process only one year:
    python scripts/02_data_pipeline_tiles.py --year 2018

    # Custom tile directories:
    python scripts/02_data_pipeline_tiles.py --tiles-2018 path/to/2018 --tiles-2024 path/to/2024

Outputs:
    data/patches_2018.npz  (X_patches, y_labels, coords, block_ids, scaler)
    data/patches_2024.npz
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds
from pyproj import Transformer
from pathlib import Path
import argparse
import gc
import sys
import glob

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import *


# --- Tile discovery ---

def discover_tiles(tile_dir):
    """Find all .tif files in the given directory, sorted."""
    tile_dir = Path(tile_dir)
    tif_files = sorted(tile_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {tile_dir}")
    print(f"  Found {len(tif_files)} tiles in {tile_dir}")
    for t in tif_files:
        size_mb = t.stat().st_size / (1024 * 1024)
        print(f"    {t.name}  ({size_mb:,.0f} MB)")
    return tif_files


# --- Training point loading (reused from original) ---

def load_training_points(csv_path):
    """
    Load training points CSV exported from GEE.
    Handles multiple GEE export formats including Point and Polygon geometries.
    """
    print(f"Loading training points: {csv_path}")
    df = pd.read_csv(csv_path)

    # Handle GEE export format (.geo column contains GeoJSON)
    if '.geo' in df.columns:
        import json

        def extract_coords(g):
            """Extract lon/lat from GeoJSON (Point or Polygon centroid)."""
            geom = json.loads(g) if isinstance(g, str) else g
            if geom['type'] == 'Point':
                return geom['coordinates'][0], geom['coordinates'][1]
            elif geom['type'] in ('Polygon', 'MultiPolygon'):
                ring = (geom['coordinates'][0] if geom['type'] == 'Polygon'
                        else geom['coordinates'][0][0])
                lons = [p[0] for p in ring]
                lats = [p[1] for p in ring]
                return sum(lons) / len(lons), sum(lats) / len(lats)
            else:
                raise ValueError(f"Unsupported geometry type: {geom['type']}")

        coords = df['.geo'].apply(extract_coords)
        df['longitude'] = coords.apply(lambda c: c[0])
        df['latitude'] = coords.apply(lambda c: c[1])

    elif 'longitude' not in df.columns:
        for lon_col in ['lon', 'x', 'X', 'Longitude']:
            if lon_col in df.columns:
                df['longitude'] = df[lon_col]
                break
        for lat_col in ['lat', 'y', 'Y', 'Latitude']:
            if lat_col in df.columns:
                df['latitude'] = df[lat_col]
                break

    # Handle GEE property name: might be 'label' or 'class'
    if 'label' not in df.columns and 'class' in df.columns:
        df['label'] = df['class']
        print("  NOTE: Renamed 'class' column to 'label'")

    print(f"  Points: {len(df)}, Classes: {df['label'].value_counts().to_dict()}")
    print(f"  Lon range: [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]")
    print(f"  Lat range: [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}]")
    return df


# --- Core helpers ---

def geo_to_pixel(transform, x, y):
    """Convert projected coordinates (CRS units) to pixel (row, col)."""
    col, row = ~transform * (x, y)
    return int(round(row)), int(round(col))


def get_tile_extent_lonlat(src):
    """Return tile extent as (lon_min, lat_min, lon_max, lat_max) in EPSG:4326."""
    h, w = src.height, src.width
    bounds = array_bounds(h, w, src.transform)  # xmin, ymin, xmax, ymax in CRS
    transformer = Transformer.from_crs(str(src.crs), "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(bounds[0], bounds[1])
    lon_max, lat_max = transformer.transform(bounds[2], bounds[3])
    return lon_min, lat_min, lon_max, lat_max


# --- Patch extraction from a single tile ---

def extract_patches_from_tile(tile_path, points_df, patch_size=15):
    """
    Open one GeoTIFF tile, filter training points that fall inside it,
    and extract patch_size × patch_size patches.

    Returns arrays of (patches, labels, coords) or empty arrays.
    """
    half = patch_size // 2

    with rasterio.open(tile_path) as src:
        crs = src.crs
        transform = src.transform
        n_bands = src.count
        h, w = src.height, src.width

        # Get tile extent in lon/lat
        lon_min, lat_min, lon_max, lat_max = get_tile_extent_lonlat(src)

        # Pre-filter points that fall inside this tile
        buf = 0.002  # ~200 m buffer
        mask = (
            (points_df['longitude'] >= lon_min + buf) &
            (points_df['longitude'] <= lon_max - buf) &
            (points_df['latitude'] >= lat_min + buf) &
            (points_df['latitude'] <= lat_max - buf)
        )
        tile_pts = points_df[mask].reset_index(drop=True)

        if len(tile_pts) == 0:
            return np.array([]), np.array([]), np.array([])

        print(f"    -> {len(tile_pts)} points inside tile extent")

        # Transformer: lon/lat → raster CRS
        transformer = Transformer.from_crs("EPSG:4326", str(crs), always_xy=True)

        patches, labels, coords = [], [], []
        skip_oob, skip_nan, skip_allzero = 0, 0, 0

        # Read the full tile data
        data = src.read()  # (bands, height, width)

    # --- iterate points (tile file is closed, data is in RAM) ---
    for _, row in tile_pts.iterrows():
        x, y = transformer.transform(row['longitude'], row['latitude'])
        r, c = geo_to_pixel(transform, x, y)

        if r - half < 0 or r + half >= h or c - half < 0 or c + half >= w:
            skip_oob += 1
            continue

        patch = data[:, r - half:r + half + 1, c - half:c + half + 1]

        if patch.shape != (n_bands, patch_size, patch_size):
            skip_oob += 1
            continue

        nan_fraction = np.sum(np.isnan(patch)) / patch.size
        if nan_fraction > 0.10:
            skip_nan += 1
            continue

        if np.all(patch == 0) or np.all(np.isnan(patch)):
            skip_allzero += 1
            continue

        patch = np.nan_to_num(patch, nan=0.0)
        patches.append(patch.transpose(1, 2, 0))  # -> (H, W, Bands)
        labels.append(int(row['label']))
        coords.append([row['longitude'], row['latitude']])

    # Free tile data
    del data
    gc.collect()

    skipped = skip_oob + skip_nan + skip_allzero
    print(f"    → Extracted {len(patches)} patches  "
          f"(skipped {skipped}: oob={skip_oob}, nan={skip_nan}, zero={skip_allzero})")

    if len(patches) == 0:
        return np.array([]), np.array([]), np.array([])

    return (np.array(patches, dtype=np.float32),
            np.array(labels),
            np.array(coords))


# --- Geometric augmentation (offline) ---

def augment_patches_geometric(patches, labels, coords):
    """
    Apply geometric augmentations to expand the dataset 8x.
    Rotations (90, 180, 270) + flips (horizontal, vertical, both)
    + the original = 8 variants per patch.
    """
    n_original = len(patches)
    aug_patches = [patches]
    aug_labels = [labels]
    aug_coords = [coords]

    for k in [1, 2, 3]:
        rotated = np.rot90(patches, k=k, axes=(1, 2))
        aug_patches.append(rotated)
        aug_labels.append(labels.copy())
        aug_coords.append(coords.copy())

    flipped_h = patches[:, :, ::-1, :]
    aug_patches.append(flipped_h)
    aug_labels.append(labels.copy())
    aug_coords.append(coords.copy())

    flipped_v = patches[:, ::-1, :, :]
    aug_patches.append(flipped_v)
    aug_labels.append(labels.copy())
    aug_coords.append(coords.copy())

    flipped_h_rot = np.rot90(flipped_h, k=1, axes=(1, 2))
    aug_patches.append(flipped_h_rot)
    aug_labels.append(labels.copy())
    aug_coords.append(coords.copy())

    flipped_v_rot = np.rot90(flipped_v, k=1, axes=(1, 2))
    aug_patches.append(flipped_v_rot)
    aug_labels.append(labels.copy())
    aug_coords.append(coords.copy())

    all_patches = np.concatenate(aug_patches, axis=0)
    all_labels = np.concatenate(aug_labels, axis=0)
    all_coords = np.concatenate(aug_coords, axis=0)

    print(f"  Geometric augmentation: {n_original} -> {len(all_patches)} patches (8x)")
    print(f"  Per-class distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")

    return all_patches, all_labels, all_coords


# --- Normalization & spatial blocks (reused from original) ---

def normalize_patches(patches, method='minmax'):
    """Normalize patches per-band. Returns normalized patches and scaler params."""
    n, h, w, b = patches.shape
    flat = patches.reshape(-1, b)

    if method == 'minmax':
        mins = flat.min(axis=0)
        maxs = flat.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        flat_norm = (flat - mins) / ranges
        scaler_params = {'method': 'minmax', 'min': mins, 'max': maxs}
    elif method == 'standard':
        means = flat.mean(axis=0)
        stds = flat.std(axis=0)
        stds[stds == 0] = 1
        flat_norm = (flat - means) / stds
        scaler_params = {'method': 'standard', 'mean': means, 'std': stds}

    return flat_norm.reshape(n, h, w, b), scaler_params


def create_spatial_blocks(coords, n_blocks=5):
    """
    Divide study area into spatial blocks for cross-validation.
    Uses longitude strips to prevent spatial leakage.
    """
    lons = coords[:, 0]
    lon_min, lon_max = lons.min(), lons.max()
    lon_edges = np.linspace(lon_min, lon_max + 1e-6, n_blocks + 1)

    block_ids = np.zeros(len(coords), dtype=int)
    for i in range(n_blocks):
        mask = (lons >= lon_edges[i]) & (lons < lon_edges[i + 1])
        block_ids[mask] = i

    for i in range(n_blocks):
        count = np.sum(block_ids == i)
        print(f"  Block {i}: {count} samples")

    return block_ids


# --- Main pipeline ---

def prepare_data_from_tiles(year, tile_dir):
    """
    Full data-preparation pipeline for one year, reading tile-by-tile.

    1. Discover all .tif tiles in tile_dir
    2. Load training points CSV
    3. For each tile -> extract patches for points inside it
    4. Concatenate, normalize, create spatial blocks
    5. Save to data/patches_{year}.npz
    """
    points_path = POINTS_2018_CSV if year == 2018 else POINTS_2024_CSV

    # --- discover tiles ---
    print(f"\nDiscovering tiles in: {tile_dir}")
    tile_paths = discover_tiles(tile_dir)

    # --- load training points ---
    points_df = load_training_points(points_path)

    # --- extract patches tile-by-tile ---
    all_patches, all_labels, all_coords = [], [], []
    points_used = set()  # track which points have already been matched

    for i, tile_path in enumerate(tile_paths):
        print(f"\n  Tile {i+1}/{len(tile_paths)}: {tile_path.name}")

        patches, labels, coords = extract_patches_from_tile(
            tile_path, points_df, PATCH_SIZE
        )

        if len(patches) == 0:
            print(f"    -> No patches from this tile")
            continue

        all_patches.append(patches)
        all_labels.append(labels)
        all_coords.append(coords)

        # Free memory after each tile
        del patches, labels, coords
        gc.collect()

    if not all_patches:
        print(f"\n  ERROR: No valid patches extracted for {year}!")
        print(f"  Check that training points fall within the tile extents.")
        return None, None, None, None

    # --- concatenate ---
    patches_all = np.concatenate(all_patches, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)
    coords_all = np.concatenate(all_coords, axis=0)
    del all_patches, all_labels, all_coords
    gc.collect()

    print(f"\n  Total patches collected: {patches_all.shape[0]}")
    print(f"  Patch shape: {patches_all.shape}")

    # --- deduplicate (a point near a tile boundary could appear in 2 tiles) ---
    print("\nDeduplicating patches (tile-overlap check)...")
    # Use coordinate rounding to detect duplicates
    coord_keys = np.round(coords_all, 6)
    _, unique_idx = np.unique(coord_keys, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)

    n_dupes = len(patches_all) - len(unique_idx)
    if n_dupes > 0:
        print(f"  Removed {n_dupes} duplicate patches from tile overlaps")
        patches_all = patches_all[unique_idx]
        labels_all = labels_all[unique_idx]
        coords_all = coords_all[unique_idx]

    print(f"  Final patch count: {patches_all.shape[0]}")
    print(f"  Labels distribution: {dict(zip(*np.unique(labels_all, return_counts=True)))}")

    # --- geometric augmentation (offline, 8x) ---
    if AUGMENT_GEOMETRIC:
        print("\nApplying geometric augmentation...")
        patches_all, labels_all, coords_all = augment_patches_geometric(
            patches_all, labels_all, coords_all
        )
        gc.collect()
    else:
        print("\nGeometric augmentation: DISABLED (set AUGMENT_GEOMETRIC=True in config)")

    # --- normalize ---
    print("\nNormalizing patches...")
    patches_norm, scaler = normalize_patches(patches_all, method='minmax')
    del patches_all
    gc.collect()

    # --- spatial blocks ---
    print("Creating spatial blocks for cross-validation...")
    block_ids = create_spatial_blocks(coords_all, N_FOLDS)

    # --- save ---
    out_path = f"{DATA_DIR}/patches_{year}.npz"
    np.savez_compressed(
        out_path,
        patches=patches_norm,
        labels=labels_all,
        coords=coords_all,
        block_ids=block_ids,
        scaler_min=scaler['min'],
        scaler_max=scaler['max'],
        band_names=BAND_NAMES,
    )
    print(f"\nSaved to {out_path}")
    print(f"  Patches shape: {patches_norm.shape}")
    print(f"  Labels distribution: {dict(zip(*np.unique(labels_all, return_counts=True)))}")

    return patches_norm, labels_all, coords_all, block_ids


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Data pipeline -- process .tif tiles from Stack_20xx folders"
    )
    parser.add_argument(
        '--year', type=int, choices=[2018, 2024], default=None,
        help='Process only this year (default: both)'
    )
    parser.add_argument(
        '--tiles-2018', type=str,
        default=f"{DATA_DIR}/Stack_2018",
        help='Directory containing 2018 tile .tif files'
    )
    parser.add_argument(
        '--tiles-2024', type=str,
        default=f"{DATA_DIR}/Stack_2024",
        help='Directory containing 2024 tile .tif files'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    Path(DATA_DIR).mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print("=" * 60)
    print("DATA PIPELINE (tile mode) -- CNN-RF Deforestation Detection")
    print("=" * 60)

    years_to_process = [args.year] if args.year else [2018, 2024]
    tile_dirs = {2018: args.tiles_2018, 2024: args.tiles_2024}

    for year in years_to_process:
        print(f"\n{'=' * 60}")
        print(f"Processing year: {year}")
        print(f"Tile directory:  {tile_dirs[year]}")
        print(f"{'=' * 60}")
        try:
            prepare_data_from_tiles(year, tile_dirs[year])
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            print(f"  Make sure tile .tif files exist in {tile_dirs[year]}")

        # Free memory between years
        gc.collect()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
