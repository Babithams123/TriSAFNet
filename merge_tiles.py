"""
merge_tiles.py
Merge split GEE GeoTIFF exports into single files.

GEE splits large exports into tiles like:
  stack_2018-0000000000-0000000000.tif
  stack_2018-0000000000-0000000001.tif
  stack_2018-0000000001-0000000000.tif
  ...

This script merges them into:
  stack_2018.tif
  stack_2024.tif

Usage:
    python scripts/merge_tiles.py

    # If tiles are in a different folder:
    python scripts/merge_tiles.py --input ./downloads --output ./data
"""

import argparse
import glob
import os
import sys
from pathlib import Path

try:
    import rasterio
    from rasterio.merge import merge
except ImportError:
    print("Installing rasterio...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rasterio"])
    import rasterio
    from rasterio.merge import merge

import gc
import numpy as np


def find_tile_groups(input_dir):
    """
    Find groups of tiles that belong to the same export.
    Groups by the base name before the tile indices.

    Examples:
      stack_2018-0000000000-0000000000.tif  →  group "stack_2018"
      stack_2018-0000000001-0000000000.tif  →  group "stack_2018"
      stack_2024-0000000000-0000000000.tif  →  group "stack_2024"
    """
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))

    if not tif_files:
        print(f"No .tif files found in {input_dir}")
        return {}

    groups = {}
    for f in tif_files:
        basename = os.path.basename(f)

        # Check if this is a tile (has -XXXX-XXXX pattern)
        # GEE pattern: name-YYYYYYYYYY-XXXXXXXXXX.tif
        parts = basename.replace('.tif', '').rsplit('-', 2)

        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            # It's a tile — group name is everything before the last two number parts
            group_name = '-'.join(parts[:-2])
        else:
            # Not a tile — it's already a single file, skip
            group_name = basename.replace('.tif', '')
            # Check if it's already merged (no tile suffix)
            if len(tif_files) == 1 or not any(group_name in os.path.basename(t) for t in tif_files if t != f):
                print(f"  {basename} — already a single file, skipping")
                continue

        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(f)

    return groups


def merge_tiles(tile_paths, output_path):
    """Merge multiple GeoTIFF tiles into a single file (memory-efficient).

    Instead of loading all tiles into RAM at once, this calculates the
    output extent from metadata, creates an empty output file on disk,
    then copies each tile band-by-band in small windows.
    """
    from rasterio.transform import array_bounds
    from rasterio.windows import from_bounds

    print(f"\n  Merging {len(tile_paths)} tiles...")

    # ── 1. Collect metadata from every tile ──────────────────────────
    tile_infos = []
    for i, t in enumerate(tile_paths):
        size_mb = os.path.getsize(t) / 1024 / 1024
        with rasterio.open(t) as src:
            print(f"    Tile {i+1}: {os.path.basename(t)} "
                  f"({src.width}x{src.height}, {src.count} bands, {size_mb:.0f} MB)")
            tile_infos.append({
                'path': t,
                'bounds': src.bounds,
                'res': src.res,
                'count': src.count,
                'dtype': src.dtypes[0],
                'crs': src.crs,
                'nodata': src.nodata,
                'profile': src.profile.copy(),
            })

    # ── 2. Calculate output extent ───────────────────────────────────
    xs = []
    ys = []
    for info in tile_infos:
        b = info['bounds']
        xs.extend([b.left, b.right])
        ys.extend([b.bottom, b.top])

    dst_left, dst_bottom = min(xs), min(ys)
    dst_right, dst_top = max(xs), max(ys)

    res_x, res_y = tile_infos[0]['res']
    out_width  = int(round((dst_right  - dst_left) / res_x))
    out_height = int(round((dst_top    - dst_bottom) / res_y))
    n_bands    = tile_infos[0]['count']

    out_transform = rasterio.transform.from_bounds(
        dst_left, dst_bottom, dst_right, dst_top, out_width, out_height
    )

    print(f"  Output dimensions: {out_width}x{out_height}, {n_bands} bands")

    # ── 3. Create the output file on disk ────────────────────────────
    out_profile = tile_infos[0]['profile'].copy()
    out_profile.update({
        'height':     out_height,
        'width':      out_width,
        'transform':  out_transform,
        'compress':   'lzw',
        'tiled':      True,
        'blockxsize': 512,
        'blockysize': 512,
        'BIGTIFF':    'YES',
    })
    if tile_infos[0]['nodata'] is not None:
        out_profile['nodata'] = tile_infos[0]['nodata']

    # ── 4. Write each tile into the output, band-by-band / window ───
    WINDOW_SIZE = 1024  # rows per read/write chunk — keep RAM usage low

    with rasterio.open(output_path, 'w', **out_profile) as dst:
        for idx, info in enumerate(tile_infos):
            print(f"    Writing tile {idx+1}/{len(tile_infos)} "
                  f"({os.path.basename(info['path'])}) ...")

            with rasterio.open(info['path']) as src:
                # Where does this tile sit inside the output grid?
                dst_window = from_bounds(
                    *src.bounds, transform=out_transform
                ).round_offsets().round_lengths()

                col_off = int(dst_window.col_off)
                row_off = int(dst_window.row_off)

                # Read / write in horizontal strips
                for row_start in range(0, src.height, WINDOW_SIZE):
                    row_end = min(row_start + WINDOW_SIZE, src.height)
                    rows = row_end - row_start

                    src_window = rasterio.windows.Window(
                        0, row_start, src.width, rows
                    )
                    dst_write_window = rasterio.windows.Window(
                        col_off, row_off + row_start, src.width, rows
                    )

                    # Read all bands for this strip
                    data = src.read(window=src_window)    # (bands, rows, cols)
                    dst.write(data, window=dst_write_window)
                    del data

    # ── 5. Free memory ───────────────────────────────────────────────
    del tile_infos
    gc.collect()

    out_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved: {output_path} ({out_size_mb:.0f} MB)")
    print(f"  Final: {out_width}x{out_height} pixels, {n_bands} bands")


def main():
    parser = argparse.ArgumentParser(description='Merge GEE split GeoTIFF tiles')
    parser.add_argument('--input', default='./data',
                        help='Directory containing tile files (default: ./data)')
    parser.add_argument('--output', default='./data',
                        help='Directory for merged output (default: ./data)')
    parser.add_argument('--delete-tiles', action='store_true',
                        help='Delete original tiles after successful merge')
    args = parser.parse_args()

    print("=" * 60)
    print("MERGE GEE GEOTIFF TILES")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    os.makedirs(args.output, exist_ok=True)

    # Find tile groups
    groups = find_tile_groups(args.input)

    if not groups:
        print("\nNo tile groups found to merge.")
        print("Make sure your .tif files are in the input directory.")
        return

    print(f"\nFound {len(groups)} tile group(s):")
    for name, tiles in groups.items():
        total_mb = sum(os.path.getsize(t) for t in tiles) / 1024 / 1024
        print(f"  {name}: {len(tiles)} tiles ({total_mb:.0f} MB total)")

    # Merge each group
    for name, tiles in groups.items():
        output_path = os.path.join(args.output, f"{name}.tif")

        # Skip if merged file already exists
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"\n  {name}: already exists ({size_mb:.0f} MB), skipping")
            continue

        if len(tiles) == 1:
            # Single tile — just rename/copy
            if os.path.abspath(tiles[0]) != os.path.abspath(output_path):
                import shutil
                shutil.copy2(tiles[0], output_path)
                print(f"\n  {name}: single tile, copied to {output_path}")
        else:
            merge_tiles(tiles, output_path)

        # Free memory between groups
        gc.collect()
        print(f"  [Memory freed after merging '{name}']")

    # Optionally delete original tiles
    if args.delete_tiles:
        print("\nDeleting original tiles...")
        for name, tiles in groups.items():
            for t in tiles:
                merged_path = os.path.join(args.output, f"{name}.tif")
                if os.path.abspath(t) != os.path.abspath(merged_path):
                    os.remove(t)
                    print(f"  Deleted: {os.path.basename(t)}")

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"\nMerged files in {args.output}/:")
    for f in sorted(Path(args.output).glob("*.tif")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.0f} MB")


if __name__ == '__main__':
    main()