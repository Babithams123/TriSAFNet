"""
07_publication_maps.py
Generate publication-ready LULC, Forest, and Deforestation maps from
GEE-exported raw classification GeoTIFFs.

Maps include all standard cartographic elements for academic papers:
  - North arrow
  - Scale bar (km)
  - Legend with class colours
  - Title with model name / year
  - Lat/Lon graticules with tick labels
  - CRS / projection info
  - Data source attribution
  - 300 DPI for print quality

Usage:
    python scripts/07_publication_maps.py
    python scripts/07_publication_maps.py --input data/gee_classified --output outputs/publication_maps
    python scripts/07_publication_maps.py --models TriSAFNet_Proposed RF_Only --types LULC_2018 LULC_2024

Expects raw GeoTIFFs from GEE with naming convention:
    RAW_<ModelSafe>_<MapType>.tif
    e.g. RAW_TriSAFNet_Proposed_LULC_2018.tif
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.crs import CRS
except ImportError:
    print("Installing rasterio...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rasterio"])
    import rasterio
    from rasterio.crs import CRS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.offsetbox import AnchoredText, AnchoredOffsetbox, TextArea, VPacker
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ── Configuration ─────────────────────────────────────────────

CLASS_NAMES = ["Forest", "Plantation", "Fallow Land", "Water Bodies", "Built-up Area"]
CLASS_COLORS = ["#00770c", "#ba02d6", "#815756", "#0821ff", "#ff0000"]
N_CLASSES = 5

FOREST_COLOR = "#00FF00"
DEFOREST_COLORS = ["#228B22", "#FF0000"]  # no-change, deforested

MODEL_DISPLAY_NAMES = {
    "TriSAFNet_Proposed": "TriSAFNet (Proposed)",
    "Lightweight_CNN_RF": "Lightweight CNN-RF",
    "CNN_Only": "CNN-Only",
    "CNN_SVM": "CNN-SVM",
    "CNN_ExtraTrees": "CNN-ExtraTrees",
    "CNN_XGBoost": "CNN-XGBoost",
    "CNN_LightGBM": "CNN-LightGBM",
    "EfficientNet_B0": "EfficientNet-B0",
    "RF_Only": "RF-Only",
    "MobileNetV2": "MobileNetV2",
}


# ── Helper Functions ──────────────────────────────────────────

def parse_filename(path: Path):
    """Parse a RAW_<model>_<type>.tif filename.
    
    Returns (model_safe, map_type) or (None, None) if not parseable.
    """
    stem = path.stem
    if not stem.startswith("RAW_"):
        return None, None
    
    rest = stem[4:]  # remove RAW_
    
    # Try to match known map types from the end
    known_types = [
        "Deforestation_2018_2024",
        "Forest_2018", "Forest_2024",
        "LULC_2018", "LULC_2024",
    ]
    for mt in known_types:
        if rest.endswith("_" + mt):
            model_safe = rest[: -(len(mt) + 1)]
            return model_safe, mt
    
    return None, None


def get_extent_latlon(src):
    """Get raster extent in lat/lon degrees."""
    bounds = src.bounds
    
    # If already in EPSG:4326, return directly
    if src.crs and src.crs.to_epsg() == 4326:
        return bounds.left, bounds.right, bounds.bottom, bounds.top
    
    # Otherwise transform bounds to WGS84
    from rasterio.warp import transform_bounds
    left, bottom, right, top = transform_bounds(
        src.crs, CRS.from_epsg(4326),
        bounds.left, bounds.bottom, bounds.right, bounds.top
    )
    return left, right, bottom, top


def add_north_arrow(ax, x=0.95, y=0.95, size=30):
    """Add a north arrow to the axes."""
    ax.annotate(
        '', xy=(x, y), xytext=(x, y - 0.06),
        xycoords='axes fraction',
        arrowprops=dict(
            arrowstyle='->', lw=2.0, color='black',
            mutation_scale=size,
        )
    )
    ax.text(
        x, y + 0.02, 'N', transform=ax.transAxes,
        fontsize=14, fontweight='bold', ha='center', va='bottom',
        path_effects=[pe.withStroke(linewidth=3, foreground='white')]
    )


def add_scale_bar(ax, src):
    """Add a scale bar in kilometres."""
    # Determine metres per pixel
    if src.crs and src.crs.is_geographic:
        # Approximate: 1 degree latitude ≈ 111,320 m
        m_per_pixel = abs(src.res[0]) * 111320
    else:
        m_per_pixel = abs(src.res[0])
    
    scalebar = ScaleBar(
        m_per_pixel, units='m', location='lower left',
        length_fraction=0.25, width_fraction=0.015,
        font_properties={'size': 10, 'weight': 'bold'},
        box_alpha=0.85, pad=0.5, border_pad=0.8,
        sep=5, color='black', box_color='white',
        scale_formatter=lambda value, unit: f"{value/1000:.0f} km"
        if value >= 1000 else f"{value:.0f} m"
    )
    ax.add_artist(scalebar)


def add_graticules(ax, extent, n_ticks=5):
    """Add lat/lon graticules with labels."""
    left, right, bottom, top = extent
    
    x_ticks = np.linspace(left, right, n_ticks)
    y_ticks = np.linspace(bottom, top, n_ticks)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{x:.1f}°E" for x in x_ticks], fontsize=8)
    ax.set_yticklabels([f"{y:.1f}°N" for y in y_ticks], fontsize=8)
    ax.tick_params(axis='both', direction='in', length=4, width=0.8)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, color='gray')


def add_attribution(fig, crs_info="EPSG:4326"):
    """Add data source and CRS info at the bottom."""
    fig.text(
        0.5, 0.01,
        f"Data: Sentinel-1/2, ALOS PALSAR-2, Landsat 8, Copernicus DEM | "
        f"CRS: {crs_info} | Karnataka, India",
        ha='center', va='bottom', fontsize=7, color='#555555',
        fontstyle='italic'
    )


# ── Map Rendering Functions ──────────────────────────────────

def render_lulc_map(tif_path, output_dir, model_name, year,
                     class_names=CLASS_NAMES, class_colors=CLASS_COLORS):
    """Render a publication-quality LULC classification map."""
    print(f"  Rendering LULC map: {model_name} — {year}")
    
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        extent = get_extent_latlon(src)
        crs_str = str(src.crs) if src.crs else "Unknown"
    
    # Mask nodata
    masked = np.ma.masked_where((data < 0) | (data >= N_CLASSES), data)
    
    # Colour map
    cmap = ListedColormap(class_colors[:N_CLASSES])
    cmap.set_bad(color='#f0f0f0')  # light gray for nodata
    norm = BoundaryNorm(boundaries=np.arange(-0.5, N_CLASSES + 0.5, 1), ncolors=N_CLASSES)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 12), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.06)
    
    im = ax.imshow(
        masked, cmap=cmap, norm=norm,
        extent=[extent[0], extent[1], extent[2], extent[3]],
        interpolation='nearest', aspect='auto'
    )
    
    # Title
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    ax.set_title(
        f"Land Use / Land Cover Classification — {year}\n{display_name}",
        fontsize=14, fontweight='bold', pad=12
    )
    
    # Graticules
    add_graticules(ax, extent)
    
    # Legend
    patches = [mpatches.Patch(facecolor=class_colors[i], edgecolor='black',
                               linewidth=0.5, label=class_names[i])
               for i in range(N_CLASSES)]
    legend = ax.legend(
        handles=patches, loc='lower right', fontsize=9,
        framealpha=0.95, edgecolor='#333333', fancybox=True,
        title='Land Cover Classes', title_fontsize=10,
        borderpad=0.8, labelspacing=0.6
    )
    legend.get_frame().set_linewidth(1.0)
    
    # North arrow
    add_north_arrow(ax)
    
    # Scale bar
    with rasterio.open(tif_path) as src:
        add_scale_bar(ax, src)
    
    # Attribution
    add_attribution(fig, crs_str)
    
    # Axis labels
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    
    # Save
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    base = f"{safe_name}_LULC_{year}"
    
    png_path = output_dir / f"{base}.png"
    pdf_path = output_dir / f"{base}.pdf"
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"    Saved: {png_path}")
    print(f"    Saved: {pdf_path}")


def render_forest_map(tif_path, output_dir, model_name, year):
    """Render a publication-quality forest cover map."""
    print(f"  Rendering Forest map: {model_name} — {year}")
    
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        extent = get_extent_latlon(src)
        crs_str = str(src.crs) if src.crs else "Unknown"
    
    # Forest mask: 1 = forest, 0 = other
    masked = np.ma.masked_where(data == 0, data)
    
    cmap = ListedColormap(['#e0e0e0', '#00AA00'])  # gray background, green forest
    cmap.set_bad(color='#e0e0e0')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 12), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.06)
    
    ax.imshow(
        data, cmap=ListedColormap(['#e8e8e8', '#00AA00']),
        extent=[extent[0], extent[1], extent[2], extent[3]],
        interpolation='nearest', aspect='auto', vmin=0, vmax=1
    )
    
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    ax.set_title(
        f"Forest Cover — {year}\n{display_name}",
        fontsize=14, fontweight='bold', pad=12
    )
    
    add_graticules(ax, extent)
    
    patches = [
        mpatches.Patch(facecolor='#00AA00', edgecolor='black', linewidth=0.5, label='Forest'),
        mpatches.Patch(facecolor='#e8e8e8', edgecolor='black', linewidth=0.5, label='Non-Forest'),
    ]
    legend = ax.legend(
        handles=patches, loc='lower right', fontsize=9,
        framealpha=0.95, edgecolor='#333333', fancybox=True,
        title='Cover Type', title_fontsize=10,
        borderpad=0.8, labelspacing=0.6
    )
    legend.get_frame().set_linewidth(1.0)
    
    add_north_arrow(ax)
    with rasterio.open(tif_path) as src:
        add_scale_bar(ax, src)
    add_attribution(fig, crs_str)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    base = f"{safe_name}_Forest_{year}"
    
    png_path = output_dir / f"{base}.png"
    pdf_path = output_dir / f"{base}.pdf"
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"    Saved: {png_path}")
    print(f"    Saved: {pdf_path}")


def render_deforestation_map(tif_path, output_dir, model_name):
    """Render a publication-quality deforestation change map."""
    print(f"  Rendering Deforestation map: {model_name}")
    
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        extent = get_extent_latlon(src)
        crs_str = str(src.crs) if src.crs else "Unknown"
    
    cmap = ListedColormap(['#228B22', '#FF0000'])  # green = no change, red = deforested
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 12), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.06)
    
    ax.imshow(
        data, cmap=cmap,
        extent=[extent[0], extent[1], extent[2], extent[3]],
        interpolation='nearest', aspect='auto', vmin=0, vmax=1
    )
    
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    ax.set_title(
        f"Deforestation Change Detection (2018 → 2024)\n{display_name}",
        fontsize=14, fontweight='bold', pad=12
    )
    
    add_graticules(ax, extent)
    
    patches = [
        mpatches.Patch(facecolor='#228B22', edgecolor='black', linewidth=0.5, label='No Change / Non-Forest'),
        mpatches.Patch(facecolor='#FF0000', edgecolor='black', linewidth=0.5, label='Deforested (2018→2024)'),
    ]
    legend = ax.legend(
        handles=patches, loc='lower right', fontsize=9,
        framealpha=0.95, edgecolor='#333333', fancybox=True,
        title='Change Type', title_fontsize=10,
        borderpad=0.8, labelspacing=0.6
    )
    legend.get_frame().set_linewidth(1.0)
    
    add_north_arrow(ax)
    with rasterio.open(tif_path) as src:
        add_scale_bar(ax, src)
    add_attribution(fig, crs_str)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    base = f"{safe_name}_Deforestation_2018_2024"
    
    png_path = output_dir / f"{base}.png"
    pdf_path = output_dir / f"{base}.pdf"
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"    Saved: {png_path}")
    print(f"    Saved: {pdf_path}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready LULC maps from GEE raw classification GeoTIFFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/07_publication_maps.py
  python scripts/07_publication_maps.py --input data/gee_classified
  python scripts/07_publication_maps.py --models TriSAFNet_Proposed RF_Only
  python scripts/07_publication_maps.py --types LULC_2018 LULC_2024 Deforestation_2018_2024
        """
    )
    parser.add_argument('--input', default='./data/gee_classified',
                        help='Directory containing RAW_*.tif files from GEE (default: ./data/gee_classified)')
    parser.add_argument('--output', default='./outputs/publication_maps',
                        help='Output directory for rendered maps (default: ./outputs/publication_maps)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Filter: only process these model safe-names (e.g. TriSAFNet_Proposed RF_Only)')
    parser.add_argument('--types', nargs='+', default=None,
                        help='Filter: only process these map types (e.g. LULC_2018 LULC_2024 Deforestation_2018_2024)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI (default: 300)')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PUBLICATION MAP GENERATOR")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    if not input_dir.exists():
        print(f"\n❌ Input directory not found: {input_dir}")
        print(f"\nTo use this script:")
        print(f"  1. Run the updated gee_karnataka_classify.js in GEE Code Editor")
        print(f"  2. Go to Tasks tab and run the RAW_* export tasks")
        print(f"  3. Download RAW_*.tif files from Google Drive to: {input_dir}")
        print(f"  4. Re-run this script")
        sys.exit(1)

    # Find all RAW_*.tif files
    tif_files = sorted(input_dir.glob("RAW_*.tif"))
    
    if not tif_files:
        print(f"\n❌ No RAW_*.tif files found in {input_dir}")
        print(f"\nMake sure files follow the naming convention:")
        print(f"  RAW_TriSAFNet_Proposed_LULC_2018.tif")
        print(f"  RAW_RF_Only_LULC_2024.tif")
        print(f"  RAW_TriSAFNet_Proposed_Deforestation_2018_2024.tif")
        sys.exit(1)

    print(f"\nFound {len(tif_files)} raw classification files:")
    
    # Parse and filter
    tasks = []
    for tif in tif_files:
        model_safe, map_type = parse_filename(tif)
        if model_safe is None:
            print(f"  ⚠ Skipping (unrecognised format): {tif.name}")
            continue
        
        if args.models and model_safe not in args.models:
            continue
        if args.types and map_type not in args.types:
            continue
        
        tasks.append((tif, model_safe, map_type))
        display = MODEL_DISPLAY_NAMES.get(model_safe, model_safe)
        print(f"  ✓ {tif.name} → {display} / {map_type}")

    if not tasks:
        print("\n❌ No matching files after filtering.")
        sys.exit(1)

    print(f"\nProcessing {len(tasks)} maps...\n")

    # Render each map
    for tif_path, model_safe, map_type in tasks:
        try:
            if map_type.startswith("LULC_"):
                year = map_type.split("_")[1]
                render_lulc_map(tif_path, output_dir, model_safe, year)
            
            elif map_type.startswith("Forest_"):
                year = map_type.split("_")[1]
                render_forest_map(tif_path, output_dir, model_safe, year)
            
            elif map_type.startswith("Deforestation"):
                render_deforestation_map(tif_path, output_dir, model_safe)
            
            else:
                print(f"  ⚠ Unknown map type: {map_type}")
        
        except Exception as e:
            print(f"  ❌ Error processing {tif_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("PUBLICATION MAP GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs in: {output_dir}")
    print(f"  PNG files: {len(list(output_dir.glob('*.png')))}")
    print(f"  PDF files: {len(list(output_dir.glob('*.pdf')))}")
    print(f"\nThese maps are 300 DPI and ready for paper submission.")


if __name__ == '__main__':
    main()
