// ============================================================
// GEE EXPORT SCRIPT — 27-Band Multi-Sensor Stack
// Sentinel-1 C-band + ALOS PALSAR-2 L-band + Sentinel-2
// (annual median composite) + Landsat 8 LST + Copernicus DEM
// + Dynamic World / ESA WorldCover extra training points
// ============================================================

// ===== YOUR EXISTING TRAINING POINTS =====
// (Import in the Code Editor as assets:
//  forest_2018, plantation_2018, fallow_2018, water_2018, builtup_2018,
//  forest_2024, plantation_2024, fallow_2024, water_2024, builtup_2024)

// Class mapping (MUST match config.py):
//   0 = Forest
//   1 = Plantation
//   2 = Fallow Land
//   3 = Water Bodies
//   4 = Built-up Area

forest_2018 = forest_2018.map(function(f) { return f.set('label', 0, 'class_name', 'Forest', 'year', 2018); });
plantation_2018 = plantation_2018.map(function(f) { return f.set('label', 1, 'class_name', 'Plantation', 'year', 2018); });
fallow_2018 = fallow_2018.map(function(f) { return f.set('label', 2, 'class_name', 'Fallow', 'year', 2018); });
water_2018 = water_2018.map(function(f) { return f.set('label', 3, 'class_name', 'Water', 'year', 2018); });
builtup_2018 = builtup_2018.map(function(f) { return f.set('label', 4, 'class_name', 'Built_up', 'year', 2018); });

var allTrainingPoints_2018 = forest_2018
  .merge(plantation_2018)
  .merge(fallow_2018)
  .merge(water_2018)
  .merge(builtup_2018);

forest_2024 = forest_2024.map(function(f) { return f.set('label', 0, 'class_name', 'Forest', 'year', 2024); });
plantation_2024 = plantation_2024.map(function(f) { return f.set('label', 1, 'class_name', 'Plantation', 'year', 2024); });
fallow_2024 = fallow_2024.map(function(f) { return f.set('label', 2, 'class_name', 'Fallow', 'year', 2024); });
water_2024 = water_2024.map(function(f) { return f.set('label', 3, 'class_name', 'Water', 'year', 2024); });
builtup_2024 = builtup_2024.map(function(f) { return f.set('label', 4, 'class_name', 'Built_up', 'year', 2024); });

var allTrainingPoints_2024 = forest_2024
  .merge(plantation_2024)
  .merge(fallow_2024)
  .merge(water_2024)
  .merge(builtup_2024);

print('=== TRAINING POINTS CHECK ===');
print('2018 total:', allTrainingPoints_2018.size());
print('2024 total:', allTrainingPoints_2024.size());

for (var c = 0; c < 5; c++) {
  var count18 = allTrainingPoints_2018.filter(ee.Filter.eq('label', c)).size();
  var count24 = allTrainingPoints_2024.filter(ee.Filter.eq('label', c)).size();
  print('Class ' + c + ': 2018=' + count18.getInfo() + ', 2024=' + count24.getInfo());
}

// ===== STUDY AREA =====
var karnataka = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM1_NAME', 'Karnataka'));

Map.centerObject(karnataka, 7);
Map.addLayer(karnataka, {color: 'red'}, 'Karnataka');


// ============================================================
// SENTINEL-2 — ANNUAL MEDIAN COMPOSITE (Jan–Dec)
// All 12 reflectance bands + 7 spectral indices
//
// Annual compositing maximises cloud-free coverage across
// Karnataka, which experiences heavy monsoon cloud cover
// (Jul–Oct) that limits seasonal compositing approaches.
// ============================================================

function maskS2clouds(image) {
  var cloudProb = image.select('MSK_CLDPRB');
  var scl = image.select('SCL');
  var cloud = cloudProb.lt(10);
  var shadow = scl.eq(3);
  var cirrus = scl.eq(10);
  var mask = cloud.and(cirrus.neq(1)).and(shadow.neq(1));
  return image.updateMask(mask).divide(10000);
}

function getS2AnnualComposite(year) {
  var startDate = year + '-01-01';
  var endDate   = year + '-12-31';

  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(karnataka)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    .map(maskS2clouds);

  var allBands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                  'B8', 'B8A', 'B9', 'B11', 'B12'];
  var composite = s2.median().select(allBands).clip(karnataka);

  var ndvi  = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndwi  = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var mndwi = composite.normalizedDifference(['B3', 'B11']).rename('MNDWI');

  var evi = composite.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': composite.select('B8'),
      'RED': composite.select('B4'),
      'BLUE': composite.select('B2')
    }).rename('EVI');

  var bsi = composite.expression(
    '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))', {
      'SWIR': composite.select('B11'),
      'RED': composite.select('B4'),
      'NIR': composite.select('B8'),
      'BLUE': composite.select('B2')
    }).rename('BSI');

  var savi = composite.expression(
    '1.5 * ((NIR - RED) / (NIR + RED + 0.5))', {
      'NIR': composite.select('B8'),
      'RED': composite.select('B4')
    }).rename('SAVI');

  var ndbi = composite.normalizedDifference(['B11', 'B8']).rename('NDBI');

  return composite.addBands([ndvi, ndwi, mndwi, evi, bsi, savi, ndbi]);
}


// ============================================================
// SENTINEL-1 C-BAND SAR
// ============================================================

function getS1Composite(year) {
  var startDate = year + '-01-01';
  var endDate   = year + '-12-31';

  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(karnataka)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select(['VV', 'VH']);

  var composite = s1.median().clip(karnataka);

  var rvi = composite.expression(
    '4 * VH / (VV + VH)', {
      'VV': ee.Image(10).pow(composite.select('VV').divide(10)),
      'VH': ee.Image(10).pow(composite.select('VH').divide(10))
    }).rename('RVI');

  return composite.addBands(rvi);
}


// ============================================================
// ALOS PALSAR-2 L-BAND SAR
// ============================================================

function getPalsarComposite(year) {
  var yearInt = parseInt(year);
  var palsar = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH')
    .filterBounds(karnataka)
    .filter(ee.Filter.calendarRange(yearInt, yearInt, 'year'));

  var count = palsar.size();
  var image = ee.Algorithms.If(
    count.gt(0),
    palsar.first().select(['HH', 'HV']),
    ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH')
      .filterBounds(karnataka)
      .sort('system:time_start', false)
      .first().select(['HH', 'HV'])
  );

  return ee.Image(image).clip(karnataka);
}


// ============================================================
// LANDSAT 8 SURFACE TEMPERATURE
// ============================================================

function getLandsatLST(year) {
  var startDate = year + '-01-01';
  var endDate   = year + '-12-31';

  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(karnataka)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUD_COVER', 20))
    .select('ST_B10');

  var lst = l8.median()
    .multiply(0.00341802).add(149.0)
    .subtract(273.15)
    .rename('LST')
    .clip(karnataka);

  return lst;
}


// ============================================================
// TOPOGRAPHIC DATA — COPERNICUS DEM GLO-30
// ============================================================

var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30')
  .select('DEM').mosaic().clip(karnataka);
var slopeLayer = ee.Terrain.slope(dem).rename('slope');
var topo = dem.rename('elevation').addBands(slopeLayer);


// ============================================================
// BUILD 27-BAND MULTI-SOURCE STACKS
// ============================================================
// Band order (27 total):
//  0: VV        1: VH         2: RVI                   — S1 C-band SAR
//  3: HH        4: HV                                   — PALSAR-2 L-band SAR
//  5: B1        6: B2         7: B3       8: B4         — S2 annual reflectance
//  9: B5       10: B6        11: B7      12: B8
// 13: B8A      14: B9        15: B11     16: B12
// 17: NDVI     18: NDWI      19: MNDWI   20: EVI       — S2 annual indices
// 21: BSI      22: SAVI      23: NDBI
// 24: LST                                               — Landsat 8 surface temp
// 25: elevation 26: slope                               — Copernicus DEM

function buildStack(year) {
  var s1     = getS1Composite(year);
  var palsar = getPalsarComposite(year);
  var s2     = getS2AnnualComposite(year);
  var lst    = getLandsatLST(year);
  return s1.addBands(palsar).addBands(s2).addBands(lst).addBands(topo);
}

var stack2018 = buildStack('2018');
var stack2024 = buildStack('2024');

print('2018 stack bands (' + stack2018.bandNames().size().getInfo() + '):', stack2018.bandNames());
print('2024 stack bands (' + stack2024.bandNames().size().getInfo() + '):', stack2024.bandNames());


// ============================================================
// SAMPLE PIXEL VALUES AT TRAINING POINTS
// ============================================================

function samplePixels(stack, points) {
  return stack.sampleRegions({
    collection: points,
    properties: ['label'],
    scale: 10,
    tileScale: 4,
    geometries: true
  });
}

var samples2018 = samplePixels(stack2018, allTrainingPoints_2018);
var samples2024 = samplePixels(stack2024, allTrainingPoints_2024);

print('Sampled 2018:', samples2018.size());
print('Sampled 2024:', samples2024.size());
print('Sample feature 2018:', samples2018.first());


// ============================================================
// DYNAMIC WORLD EXTRA TRAINING POINTS (year-specific)
// 3000 stratified points per class across Karnataka, per year.
// Uses annual mode composites so labels reflect actual land
// cover in 2018 and 2024 respectively.
//
// Cross-validated with ESA WorldCover to retain only locations
// where both products agree — high-confidence labels.
//
// Dynamic World class mapping:
//   0 (water)              -> 3 (Water Bodies)
//   1 (trees)              -> 0 (Forest)
//   2 (grass)              -> 2 (Fallow Land)
//   3 (flooded_vegetation) -> 3 (Water Bodies)
//   4 (crops)              -> 1 (Plantation)
//   5 (shrub_and_scrub)    -> 2 (Fallow Land)
//   6 (built)              -> 4 (Built-up Area)
//   7 (bare)               -> 2 (Fallow Land)
//   8 (snow_and_ice)       -> 3 (Water Bodies)
// ============================================================

var worldcover = ee.ImageCollection('ESA/WorldCover/v200').mosaic();
var wcRemapped = worldcover.remap(
  [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
  [ 0,  2,  2,  1,  4,  2,  3,  3,  3,  0,   2]
).rename('wc_label').clip(karnataka);

function getDynamicWorldMode(year) {
  var startDate = year + '-01-01';
  var endDate   = year + '-12-31';

  var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterBounds(karnataka)
    .filterDate(startDate, endDate)
    .select('label');

  return dw.reduce(ee.Reducer.mode()).rename('dw_label').clip(karnataka);
}

function remapDynamicWorld(dwMode) {
  return dwMode.remap(
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [3, 0, 2, 3, 1, 2, 4, 2, 3]
  ).rename('label');
}

function generateExtraPoints(year, seed) {
  var dwMode = getDynamicWorldMode(year);
  var dwRemapped = remapDynamicWorld(dwMode);

  var agreement = dwRemapped.eq(wcRemapped);
  var highConfidence = dwRemapped.updateMask(agreement).clip(karnataka);

  var points = highConfidence.stratifiedSample({
    numPoints: 3000,
    classBand: 'label',
    region: karnataka,
    scale: 10,
    geometries: true,
    seed: seed
  });

  return points;
}

var extraPoints2018 = generateExtraPoints('2018', 42);
var extraPoints2024 = generateExtraPoints('2024', 43);

print('=== DYNAMIC WORLD EXTRA POINTS (2018) ===');
print('Total:', extraPoints2018.size());
for (var ec18 = 0; ec18 < 5; ec18++) {
  print('Class ' + ec18 + ':',
    extraPoints2018.filter(ee.Filter.eq('label', ec18)).size());
}

print('=== DYNAMIC WORLD EXTRA POINTS (2024) ===');
print('Total:', extraPoints2024.size());
for (var ec24 = 0; ec24 < 5; ec24++) {
  print('Class ' + ec24 + ':',
    extraPoints2024.filter(ee.Filter.eq('label', ec24)).size());
}


// ============================================================
// EXPORTS — Run ALL from the Tasks tab
// ============================================================

// --- 1. Full 27-band raster stacks (for inference / map generation) ---
Export.image.toDrive({
  image: stack2018.toFloat(),
  description: 'stack_2018',
  folder: 'CNN_RF_Data',
  region: karnataka,
  scale: 10,
  crs: 'EPSG:32643',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: stack2024.toFloat(),
  description: 'stack_2024',
  folder: 'CNN_RF_Data',
  region: karnataka,
  scale: 10,
  crs: 'EPSG:32643',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// --- 2. Training samples CSV (27-band pixel values + label + coords) ---
Export.table.toDrive({
  collection: samples2018,
  description: 'training_samples_2018',
  folder: 'CNN_RF_Data',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: samples2024,
  description: 'training_samples_2024',
  folder: 'CNN_RF_Data',
  fileFormat: 'CSV'
});

// --- 3. Training point locations (coords + label, for patch extraction) ---
Export.table.toDrive({
  collection: allTrainingPoints_2018,
  description: 'training_points_2018',
  folder: 'CNN_RF_Data',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: allTrainingPoints_2024,
  description: 'training_points_2024',
  folder: 'CNN_RF_Data',
  fileFormat: 'CSV'
});

// --- 4. Dynamic World extra training points (year-specific, 3000/class) ---
Export.table.toDrive({
  collection: extraPoints2018,
  description: 'extra_training_points_2018',
  folder: 'CNN_RF_Data',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: extraPoints2024,
  description: 'extra_training_points_2024',
  folder: 'CNN_RF_Data',
  fileFormat: 'CSV'
});


// ============================================================
// VISUAL CHECK
// ============================================================

var visRGB = {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3};
Map.addLayer(stack2018.select(['B4', 'B3', 'B2']), visRGB, 'S2 RGB 2018');
Map.addLayer(stack2018.select('NDVI'), {min: 0, max: 1, palette: ['red', 'yellow', 'green']}, 'NDVI 2018');
Map.addLayer(allTrainingPoints_2018, {color: 'white'}, 'Training Points 2018');

print('=== ALL EXPORT TASKS READY ===');
print('Go to the Tasks tab and click RUN for each task.');
print('');
print('Expected 8 exports:');
print('  1. stack_2018                    (GeoTIFF, 27 bands)');
print('  2. stack_2024                    (GeoTIFF, 27 bands)');
print('  3. training_samples_2018         (CSV)');
print('  4. training_samples_2024         (CSV)');
print('  5. training_points_2018          (CSV)');
print('  6. training_points_2024          (CSV)');
print('  7. extra_training_points_2018    (CSV, DynamicWorld ~15K points)');
print('  8. extra_training_points_2024    (CSV, DynamicWorld ~15K points)');
print('');
print('After all exports finish, download from Google Drive/CNN_RF_Data/');
print('and place files in the ./data/ folder of the Python pipeline.');
