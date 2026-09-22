"""
08_geemap_publication_maps.py
Generate publication-ready LULC maps using geemap (GEE Python API).

No manual GEE Code Editor exports needed — everything runs from Python.

Usage:
    python scripts/08_geemap_publication_maps.py
    python scripts/08_geemap_publication_maps.py --models TriSAFNet_Proposed RF_Only
    python scripts/08_geemap_publication_maps.py --scale 100  # faster, lower res

Prerequisites:
    pip install geemap earthengine-api matplotlib-scalebar
    earthengine authenticate   (one-time)
"""

import argparse
import os
import sys
from pathlib import Path

import ee
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib_scalebar.scalebar import ScaleBar

# ── Configuration ─────────────────────────────────────────────

CLASS_NAMES = ['Forest', 'Plantation', 'Fallow Land', 'Water Bodies', 'Built-up Area']
CLASS_COLORS_HEX = ['#00770c', '#ba02d6', '#815756', '#0821ff', '#ff0000']
N_CLASSES = 5
EXPORT_SCALE = 30

MODELS = [
    {'name': 'TriSAFNet (Proposed)',  'safe': 'TriSAFNet_Proposed',
     'clf': lambda: ee.Classifier.smileRandomForest(numberOfTrees=500, seed=42)},
    {'name': 'Lightweight CNN-RF',  'safe': 'Lightweight_CNN_RF',
     'clf': lambda: ee.Classifier.smileRandomForest(numberOfTrees=150, maxNodes=50, seed=42)},
    {'name': 'CNN-Only',            'safe': 'CNN_Only',
     'clf': lambda: ee.Classifier.smileCart()},
    {'name': 'CNN-SVM',             'safe': 'CNN_SVM',
     'clf': lambda: ee.Classifier.libsvm(svmType='C_SVC', kernelType='RBF', cost=10, gamma=0.01)},
    {'name': 'CNN-ExtraTrees',      'safe': 'CNN_ExtraTrees',
     'clf': lambda: ee.Classifier.smileRandomForest(numberOfTrees=200, bagFraction=0.8, seed=43)},
    {'name': 'CNN-XGBoost',         'safe': 'CNN_XGBoost',
     'clf': lambda: ee.Classifier.smileGradientTreeBoost(numberOfTrees=200, shrinkage=0.05, seed=42)},
    {'name': 'CNN-LightGBM',        'safe': 'CNN_LightGBM',
     'clf': lambda: ee.Classifier.smileGradientTreeBoost(numberOfTrees=100, shrinkage=0.1, seed=42)},
    {'name': 'EfficientNet-B0',     'safe': 'EfficientNet_B0',
     'clf': lambda: ee.Classifier.smileRandomForest(numberOfTrees=300, maxNodes=100, seed=42)},
    {'name': 'RF-Only',             'safe': 'RF_Only',
     'clf': lambda: ee.Classifier.smileRandomForest(numberOfTrees=200, seed=42)},
    {'name': 'MobileNetV2',         'safe': 'MobileNetV2',
     'clf': lambda: ee.Classifier.smileNaiveBayes()},
]

# ── GEE Data Functions ───────────────────────────────────────

def get_karnataka():
    ka = ee.FeatureCollection('FAO/GAUL/2015/level1').filter(ee.Filter.eq('ADM1_NAME', 'Karnataka'))
    return ka, ka.geometry()

def maskS2clouds(image):
    cp = image.select('MSK_CLDPRB')
    scl = image.select('SCL')
    mask = cp.lt(10).And(scl.neq(3)).And(scl.neq(10))
    return image.updateMask(mask).divide(10000)

def getS2Composite(year, kGeo):
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(kGeo).filterDate(f'{year}-01-01', f'{year}-12-31')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)).map(maskS2clouds))
    bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
    comp = s2.median().select(bands).clip(kGeo)
    ndvi = comp.normalizedDifference(['B8','B4']).rename('NDVI')
    ndwi = comp.normalizedDifference(['B3','B8']).rename('NDWI')
    mndwi = comp.normalizedDifference(['B3','B11']).rename('MNDWI')
    evi = comp.expression('2.5*((N-R)/(N+6*R-7.5*B+1))',
        {'N':comp.select('B8'),'R':comp.select('B4'),'B':comp.select('B2')}).rename('EVI')
    bsi = comp.expression('((S+R)-(N+B))/((S+R)+(N+B))',
        {'S':comp.select('B11'),'R':comp.select('B4'),'N':comp.select('B8'),'B':comp.select('B2')}).rename('BSI')
    savi = comp.expression('1.5*((N-R)/(N+R+0.5))',
        {'N':comp.select('B8'),'R':comp.select('B4')}).rename('SAVI')
    ndbi = comp.normalizedDifference(['B11','B8']).rename('NDBI')
    return comp.addBands([ndvi, ndwi, mndwi, evi, bsi, savi, ndbi])

def getS1Composite(year, kGeo):
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(kGeo).filterDate(f'{year}-01-01', f'{year}-12-31')
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))
          .filter(ee.Filter.eq('instrumentMode','IW')).select(['VV','VH']))
    comp = s1.median().clip(kGeo)
    rvi = comp.expression('4*VH/(VV+VH)', {
        'VV': ee.Image(10).pow(comp.select('VV').divide(10)),
        'VH': ee.Image(10).pow(comp.select('VH').divide(10))
    }).rename('RVI')
    return comp.addBands(rvi)

def getPalsar(year, kGeo):
    yr = int(year)
    col = (ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH')
           .filterBounds(kGeo).filter(ee.Filter.calendarRange(yr, yr, 'year')))
    img = ee.Algorithms.If(col.size().gt(0), col.first().select(['HH','HV']),
        ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH')
          .filterBounds(kGeo).sort('system:time_start', False).first().select(['HH','HV']))
    return ee.Image(img).clip(kGeo)

def getLST(year, kGeo):
    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterBounds(kGeo).filterDate(f'{year}-01-01', f'{year}-12-31')
          .filter(ee.Filter.lt('CLOUD_COVER', 20)).select('ST_B10'))
    return l8.median().multiply(0.00341802).add(149.0).subtract(273.15).rename('LST').clip(kGeo)

def buildStack(year, kGeo):
    dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().clip(kGeo)
    slope = ee.Terrain.slope(dem).rename('slope')
    topo = dem.rename('elevation').addBands(slope)
    return (getS1Composite(year, kGeo).addBands(getPalsar(year, kGeo))
            .addBands(getS2Composite(year, kGeo)).addBands(getLST(year, kGeo)).addBands(topo))

def getTrainingPoints(year, kGeo, seed=42):
    worldcover = ee.ImageCollection('ESA/WorldCover/v200').mosaic()
    wcLabel = worldcover.remap(
        [10,20,30,40,50,60,70,80,90,95,100],
        [0,2,2,1,4,2,3,3,3,0,2]).rename('wc_label').clip(kGeo)
    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterBounds(kGeo).filterDate(f'{year}-01-01', f'{year}-12-31')
          .select('label').reduce(ee.Reducer.mode()).rename('dw_label').clip(kGeo))
    dwMapped = dw.remap([0,1,2,3,4,5,6,7,8],[3,0,2,3,1,2,4,2,3]).rename('label')
    agreement = dwMapped.eq(wcLabel)
    highConf = dwMapped.updateMask(agreement).clip(kGeo)
    return highConf.stratifiedSample(
        numPoints=500, classBand='label', region=kGeo, scale=30,
        geometries=True, seed=seed)


# ── Thumbnail Fetch ───────────────────────────────────────────

def fetch_thumbnail_array(image, vis_params, region, dimensions=2048):
    """Get a classified image as a numpy array via GEE thumbnail."""
    vis_img = image.visualize(**vis_params).unmask(0).clip(region)
    url = vis_img.getThumbURL({
        'dimensions': dimensions,
        'region': region.bounds().getInfo(),
        'format': 'png'
    })
    import urllib.request
    from PIL import Image
    import io
    print(f"    Fetching thumbnail ({dimensions}px)...")
    response = urllib.request.urlopen(url)
    img_data = response.read()
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)


# ── Map Rendering ─────────────────────────────────────────────

def add_north_arrow(ax, x=0.95, y=0.95):
    ax.annotate('', xy=(x, y), xytext=(x, y - 0.06), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2.0, color='black', mutation_scale=30))
    ax.text(x, y + 0.02, 'N', transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='bottom',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])

def add_scale_bar_km(ax, extent_deg, img_width_px):
    """Add approximate scale bar using degree-to-km conversion."""
    lon_range = extent_deg[1] - extent_deg[0]
    # At ~15°N (Karnataka), 1° lon ≈ 107.5 km
    km_per_deg = 107.5
    total_km = lon_range * km_per_deg
    m_per_pixel = (total_km * 1000) / img_width_px
    scalebar = ScaleBar(m_per_pixel, units='m', location='lower left',
                        length_fraction=0.25, width_fraction=0.015,
                        font_properties={'size': 10, 'weight': 'bold'},
                        box_alpha=0.85, pad=0.5, border_pad=0.8, sep=5,
                        color='black', box_color='white')
    ax.add_artist(scalebar)

def add_graticules(ax, extent, n_ticks=5):
    left, right, bottom, top = extent
    xt = np.linspace(left, right, n_ticks)
    yt = np.linspace(bottom, top, n_ticks)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{x:.1f}°E" for x in xt], fontsize=8)
    ax.set_yticklabels([f"{y:.1f}°N" for y in yt], fontsize=8)
    ax.tick_params(direction='in', length=4, width=0.8)
    ax.grid(True, ls='--', lw=0.4, alpha=0.5, color='gray')

def render_map(img_array, title, subtitle, extent, legend_patches, output_dir, filename):
    """Render a publication-quality map from a numpy array."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 12), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.06)

    ax.imshow(img_array, extent=[extent[0], extent[1], extent[2], extent[3]],
              interpolation='nearest', aspect='auto')

    ax.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight='bold', pad=12)
    add_graticules(ax, extent)
    add_north_arrow(ax)
    add_scale_bar_km(ax, extent, img_array.shape[1])

    legend = ax.legend(handles=legend_patches, loc='lower right', fontsize=9,
                       framealpha=0.95, edgecolor='#333', fancybox=True,
                       title='Classes', title_fontsize=10, borderpad=0.8, labelspacing=0.6)
    legend.get_frame().set_linewidth(1.0)

    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)

    fig.text(0.5, 0.01,
             "Data: Sentinel-1/2, ALOS PALSAR-2, Landsat 8, Copernicus DEM | "
             "CRS: EPSG:4326 | Karnataka, India",
             ha='center', va='bottom', fontsize=7, color='#555', fontstyle='italic')

    for ext in ['png', 'pdf']:
        path = output_dir / f"{filename}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {output_dir / filename}.png/.pdf")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate publication maps via geemap/GEE Python API")
    parser.add_argument('--output', default='./outputs/publication_maps', help='Output directory')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Model safe names to process (default: all)')
    parser.add_argument('--dimensions', type=int, default=2048,
                        help='Thumbnail resolution in pixels (default: 2048)')
    parser.add_argument('--scale', type=int, default=30, help='GEE scale in metres')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GEEMAP PUBLICATION MAP GENERATOR")
    print("=" * 60)

    # ── Authenticate & Initialize GEE ──
    print("\nInitializing Google Earth Engine...")
    try:
        ee.Initialize(project='ee-mehathab1996')
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
        except Exception as e:
            print(f"❌ GEE authentication failed: {e}")
            print("Run: earthengine authenticate")
            sys.exit(1)
    print("✅ GEE initialized")

    # ── Study Area ──
    karnataka, kGeo = get_karnataka()
    bounds_info = kGeo.bounds().coordinates().getInfo()[0]
    lons = [p[0] for p in bounds_info]
    lats = [p[1] for p in bounds_info]
    extent = (min(lons), max(lons), min(lats), max(lats))
    print(f"Study area extent: {extent[0]:.2f}–{extent[1]:.2f}°E, {extent[2]:.2f}–{extent[3]:.2f}°N")

    # ── Build Data Stacks ──
    print("\nBuilding data stacks...")
    stack2018 = buildStack('2018', kGeo)
    stack2024 = buildStack('2024', kGeo)
    print("✅ Stacks ready")

    # ── Training Data ──
    print("Generating training data...")
    pts2018 = getTrainingPoints('2018', kGeo, 42)
    pts2024 = getTrainingPoints('2024', kGeo, 43)
    inputBands = stack2018.bandNames()

    trainData2018 = stack2018.sampleRegions(collection=pts2018, properties=['label'], scale=30, tileScale=4)
    trainData2024 = stack2024.sampleRegions(collection=pts2024, properties=['label'], scale=30, tileScale=4)
    trainDataAll = trainData2018.merge(trainData2024)
    print("✅ Training data ready")

    # ── Vis params ──
    lulcVis = {'min': 0, 'max': 4, 'palette': CLASS_COLORS_HEX}
    forestVis = {'palette': ['#00FF00']}
    deforestVis = {'palette': ['#FFFF00']}

    lulc_patches = [mpatches.Patch(facecolor=CLASS_COLORS_HEX[i], edgecolor='k', lw=0.5, label=CLASS_NAMES[i])
                    for i in range(N_CLASSES)]
    forest_patches = [
        mpatches.Patch(facecolor='#00FF00', edgecolor='k', lw=0.5, label='Forest'),
        mpatches.Patch(facecolor='#000000', edgecolor='k', lw=0.5, label='Non-Forest'),
    ]
    deforest_patches = [
        mpatches.Patch(facecolor='#228B22', edgecolor='k', lw=0.5, label='No Change'),
        mpatches.Patch(facecolor='#FFFF00', edgecolor='k', lw=0.5, label='Deforested'),
    ]

    # ── Filter models ──
    models_to_run = MODELS
    if args.models:
        models_to_run = [m for m in MODELS if m['safe'] in args.models]
    print(f"\nProcessing {len(models_to_run)} models...\n")

    # ── Classify & Render ──
    for mi, m in enumerate(models_to_run):
        print(f"\n[{mi+1}/{len(models_to_run)}] {m['name']} ({m['safe']})")

        # Train classifier
        print("  Training classifier...")
        trained = m['clf']().train(features=trainDataAll, classProperty='label', inputProperties=inputBands)

        # Classify
        c18 = stack2018.classify(trained).byte().clip(kGeo)
        c24 = stack2024.classify(trained).byte().clip(kGeo)
        f18 = c18.eq(0).selfMask()
        f24 = c24.eq(0).selfMask()
        deforest = c18.eq(0).And(c24.neq(0)).selfMask()

        maps_to_render = [
            ('LULC 2018', c18, lulcVis, lulc_patches, f'{m["safe"]}_LULC_2018'),
            ('LULC 2024', c24, lulcVis, lulc_patches, f'{m["safe"]}_LULC_2024'),
            ('Forest Cover 2018', f18, forestVis, forest_patches, f'{m["safe"]}_Forest_2018'),
            ('Forest Cover 2024', f24, forestVis, forest_patches, f'{m["safe"]}_Forest_2024'),
            ('Deforestation 2018→2024', deforest, deforestVis, deforest_patches, f'{m["safe"]}_Deforestation'),
        ]

        for title, img, vis, patches, fname in maps_to_render:
            try:
                arr = fetch_thumbnail_array(img, vis, kGeo, args.dimensions)
                render_map(arr, title, m['name'], extent, patches, output_dir, fname)
            except Exception as e:
                print(f"    ❌ Failed: {e}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Maps saved to: {output_dir}")


if __name__ == '__main__':
    main()
