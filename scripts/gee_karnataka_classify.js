/*
=============================================================================
  GEE CODE EDITOR: Karnataka LULC Classification — 10-Model Comparison
  
  Replicates the CNN-RF pipeline using GEE built-in classifiers.
  Uses the SAME datasets: Sentinel-1, Sentinel-2, ALOS PALSAR-2,
  Landsat 8 LST, Copernicus DEM (27 bands total).
  
  Training data: Dynamic World + ESA WorldCover agreement points.
  
  INSTRUCTIONS:
    1. Paste this script into GEE Code Editor
    2. Click Run — wait for training to complete
    3. Use the Model Selector panel (left) to switch models
    4. Use the Download Panel to get PNG links
    5. Go to Tasks tab to run Drive exports (50 tasks)
=============================================================================
*/

// ============================================================
// SECTION 1: CONFIGURATION
// ============================================================

var CLASS_NAMES = ['Forest', 'Plantation', 'Fallow Land', 'Water Bodies', 'Built-up Area'];
var CLASS_COLORS = ['#00770c', '#ba02d6', '#815756', '#0821ff', '#ff0000'];
var N_CLASSES = 5;
var FOREST_CLASS = 0;
var EXPORT_SCALE = 30;        // 30m for manageable exports; change to 10 for full-res
var EXPORT_FOLDER = 'CNN_RF_Karnataka_Maps';
var THUMB_DIM = 512;          // Keep small to avoid browser freeze; increase to 1024 if needed
var TRAIN_PTS_PER_CLASS = 500;  // Reduced for faster training; increase to 1000 if needed

// ============================================================
// SECTION 2: STUDY AREA
// ============================================================

var karnataka = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM1_NAME', 'Karnataka'));
var kGeo = karnataka.geometry();

Map.centerObject(karnataka, 7);
Map.setOptions('SATELLITE');

// ============================================================
// SECTION 3: SENTINEL-2 ANNUAL COMPOSITE
// ============================================================

function maskS2clouds(image) {
  var cp = image.select('MSK_CLDPRB');
  var scl = image.select('SCL');
  var mask = cp.lt(10).and(scl.neq(3)).and(scl.neq(10));
  return image.updateMask(mask).divide(10000);
}

function getS2Composite(year) {
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(kGeo).filterDate(year + '-01-01', year + '-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    .map(maskS2clouds);
  var bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'];
  var comp = s2.median().select(bands).clip(kGeo);
  var ndvi = comp.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndwi = comp.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var mndwi = comp.normalizedDifference(['B3', 'B11']).rename('MNDWI');
  var evi = comp.expression('2.5*((N-R)/(N+6*R-7.5*B+1))',
    { N: comp.select('B8'), R: comp.select('B4'), B: comp.select('B2') }).rename('EVI');
  var bsi = comp.expression('((S+R)-(N+B))/((S+R)+(N+B))',
    { S: comp.select('B11'), R: comp.select('B4'), N: comp.select('B8'), B: comp.select('B2') }).rename('BSI');
  var savi = comp.expression('1.5*((N-R)/(N+R+0.5))',
    { N: comp.select('B8'), R: comp.select('B4') }).rename('SAVI');
  var ndbi = comp.normalizedDifference(['B11', 'B8']).rename('NDBI');
  return comp.addBands([ndvi, ndwi, mndwi, evi, bsi, savi, ndbi]);
}

// ============================================================
// SECTION 4: SENTINEL-1 C-BAND SAR
// ============================================================

function getS1Composite(year) {
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(kGeo).filterDate(year + '-01-01', year + '-12-31')
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW')).select(['VV', 'VH']);
  var comp = s1.median().clip(kGeo);
  var rvi = comp.expression('4*VH/(VV+VH)', {
    VV: ee.Image(10).pow(comp.select('VV').divide(10)),
    VH: ee.Image(10).pow(comp.select('VH').divide(10))
  }).rename('RVI');
  return comp.addBands(rvi);
}

// ============================================================
// SECTION 5: ALOS PALSAR-2 L-BAND SAR
// ============================================================

function getPalsar(year) {
  var yr = parseInt(year);
  var col = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH')
    .filterBounds(kGeo).filter(ee.Filter.calendarRange(yr, yr, 'year'));
  var img = ee.Algorithms.If(col.size().gt(0), col.first().select(['HH', 'HV']),
    ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH')
      .filterBounds(kGeo).sort('system:time_start', false).first().select(['HH', 'HV']));
  return ee.Image(img).clip(kGeo);
}

// ============================================================
// SECTION 6: LANDSAT 8 LST
// ============================================================

function getLST(year) {
  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(kGeo).filterDate(year + '-01-01', year + '-12-31')
    .filter(ee.Filter.lt('CLOUD_COVER', 20)).select('ST_B10');
  return l8.median().multiply(0.00341802).add(149.0).subtract(273.15)
    .rename('LST').clip(kGeo);
}

// ============================================================
// SECTION 7: TOPOGRAPHIC DATA
// ============================================================

var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().clip(kGeo);
var slope = ee.Terrain.slope(dem).rename('slope');
var topo = dem.rename('elevation').addBands(slope);

// ============================================================
// SECTION 8: BUILD 27-BAND STACKS
// ============================================================

function buildStack(year) {
  return getS1Composite(year).addBands(getPalsar(year))
    .addBands(getS2Composite(year)).addBands(getLST(year)).addBands(topo);
}

print('Building data stacks...');
var stack2018 = buildStack('2018');
var stack2024 = buildStack('2024');
print('2018 bands:', stack2018.bandNames());
print('2024 bands:', stack2024.bandNames());

// ============================================================
// SECTION 9: TRAINING DATA (Dynamic World + WorldCover)
// ============================================================

var worldcover = ee.ImageCollection('ESA/WorldCover/v200').mosaic();
var wcLabel = worldcover.remap(
  [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
  [0, 2, 2, 1, 4, 2, 3, 3, 3, 0, 2]).rename('wc_label').clip(kGeo);

function getTrainingPoints(year, seed) {
  var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterBounds(kGeo).filterDate(year + '-01-01', year + '-12-31')
    .select('label').reduce(ee.Reducer.mode()).rename('dw_label').clip(kGeo);
  var dwMapped = dw.remap([0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 0, 2, 3, 1, 2, 4, 2, 3]).rename('label');
  var agreement = dwMapped.eq(wcLabel);
  var highConf = dwMapped.updateMask(agreement).clip(kGeo);
  return highConf.stratifiedSample({
    numPoints: TRAIN_PTS_PER_CLASS, classBand: 'label',
    region: kGeo, scale: 30, geometries: true, seed: seed
  });
}

print('Generating training data...');
var trainPts2018 = getTrainingPoints('2018', 42);
var trainPts2024 = getTrainingPoints('2024', 43);
print('Training pts 2018:', trainPts2018.size());
print('Training pts 2024:', trainPts2024.size());

// Sample band values at training points
var trainData2018 = stack2018.sampleRegions({
  collection: trainPts2018, properties: ['label'], scale: 30, tileScale: 4
});
var trainData2024 = stack2024.sampleRegions({
  collection: trainPts2024, properties: ['label'], scale: 30, tileScale: 4
});

// Merge both years for robust training
var trainDataAll = trainData2018.merge(trainData2024);
print('Total training samples:', trainDataAll.size());

// ============================================================
// SECTION 10: DEFINE 10 CLASSIFIERS
// ============================================================

var inputBands = stack2018.bandNames();

var MODELS = [
  {
    name: 'TriSAFNet (Proposed)',
    safe: 'TriSAFNet_Proposed',
    clf: ee.Classifier.smileRandomForest({ numberOfTrees: 500, seed: 42 })
  },
  {
    name: 'Lightweight CNN-RF',
    safe: 'Lightweight_CNN_RF',
    clf: ee.Classifier.smileRandomForest({ numberOfTrees: 150, maxNodes: 50, seed: 42 })
  },
  {
    name: 'CNN-Only',
    safe: 'CNN_Only',
    clf: ee.Classifier.smileCart()
  },
  {
    name: 'CNN-SVM',
    safe: 'CNN_SVM',
    clf: ee.Classifier.libsvm({ svmType: 'C_SVC', kernelType: 'RBF', cost: 10, gamma: 0.01 })
  },
  {
    name: 'CNN-ExtraTrees',
    safe: 'CNN_ExtraTrees',
    clf: ee.Classifier.smileRandomForest({ numberOfTrees: 200, bagFraction: 0.8, seed: 43 })
  },
  {
    name: 'CNN-XGBoost',
    safe: 'CNN_XGBoost',
    clf: ee.Classifier.smileGradientTreeBoost({ numberOfTrees: 200, shrinkage: 0.05, seed: 42 })
  },
  {
    name: 'CNN-LightGBM',
    safe: 'CNN_LightGBM',
    clf: ee.Classifier.smileGradientTreeBoost({ numberOfTrees: 100, shrinkage: 0.1, seed: 42 })
  },
  {
    name: 'EfficientNet-B0',
    safe: 'EfficientNet_B0',
    clf: ee.Classifier.smileRandomForest({ numberOfTrees: 300, maxNodes: 100, seed: 42 })
  },
  {
    name: 'RF-Only',
    safe: 'RF_Only',
    clf: ee.Classifier.smileRandomForest({ numberOfTrees: 200, seed: 42 })
  },
  {
    name: 'MobileNetV2',
    safe: 'MobileNetV2',
    clf: ee.Classifier.smileNaiveBayes()
  }
];

// ============================================================
// SECTION 11: ON-DEMAND MODEL TRAINING & CLASSIFICATION
// ============================================================
// Models are computed LAZILY — only when selected in the UI.
// This avoids the 5-minute interactive timeout.
// Results are cached so each model is computed only once.

var resultsCache = {};

function getModelResults(safeName) {
  if (resultsCache[safeName]) {
    return resultsCache[safeName];
  }

  var model = null;
  MODELS.forEach(function (m) { if (m.safe === safeName) model = m; });
  if (!model) return null;

  var trained = model.clf.train({
    features: trainDataAll,
    classProperty: 'label',
    inputProperties: inputBands
  });

  var c18 = stack2018.classify(trained).byte().clip(kGeo);
  var c24 = stack2024.classify(trained).byte().clip(kGeo);
  var f18 = c18.eq(FOREST_CLASS).selfMask();
  var f24 = c24.eq(FOREST_CLASS).selfMask();
  var deforest = c18.eq(FOREST_CLASS).and(c24.neq(FOREST_CLASS)).selfMask();

  resultsCache[safeName] = {
    name: model.name, safe: model.safe,
    lulc2018: c18, lulc2024: c24,
    forest2018: f18, forest2024: f24,
    deforestation: deforest
  };

  return resultsCache[safeName];
}

print('✅ Models ready — select a model from the panel to view it.');
print('   (Each model is computed on first selection, then cached)');

// ============================================================
// SECTION 12: VISUALIZATION PARAMETERS
// ============================================================

var lulcVis = { min: 0, max: 4, palette: CLASS_COLORS };
var forestVis = { palette: ['#00FF00'] };
var deforestVis = { palette: ['#FFFF00'] };

// ============================================================
// SECTION 13: LEGEND PANEL
// ============================================================

var legend = ui.Panel({ style: { position: 'bottom-left', padding: '8px 15px' } });

legend.add(ui.Label({
  value: 'Land Cover Classification',
  style: { fontWeight: 'bold', fontSize: '14px', margin: '0 0 6px 0' }
}));

CLASS_NAMES.forEach(function (name, i) {
  legend.add(ui.Panel({
    widgets: [
      ui.Label({
        style: {
          backgroundColor: CLASS_COLORS[i],
          padding: '8px', margin: '0 4px 4px 0', border: '1px solid #999'
        }
      }),
      ui.Label({ value: name, style: { margin: '0 0 4px 0' } })
    ],
    layout: ui.Panel.Layout.Flow('horizontal')
  }));
});

// Deforestation entry
legend.add(ui.Panel({
  widgets: [
    ui.Label({
      style: {
        backgroundColor: '#FFFF00',
        padding: '8px', margin: '0 4px 4px 0', border: '1px solid #999'
      }
    }),
    ui.Label({ value: 'Deforestation (2018→2024)', style: { margin: '0 0 4px 0' } })
  ],
  layout: ui.Panel.Layout.Flow('horizontal')
}));

// Forest entry
legend.add(ui.Panel({
  widgets: [
    ui.Label({
      style: {
        backgroundColor: '#00FF00',
        padding: '8px', margin: '0 4px 4px 0', border: '1px solid #999'
      }
    }),
    ui.Label({ value: 'Forest Cover', style: { margin: '0 0 4px 0' } })
  ],
  layout: ui.Panel.Layout.Flow('horizontal')
}));

Map.add(legend);

// ============================================================
// SECTION 14: NORTH ARROW
// ============================================================

var northArrow = ui.Panel({
  widgets: [
    ui.Label('▲', { fontSize: '22px', fontWeight: 'bold', textAlign: 'center', margin: '0' }),
    ui.Label('N', { fontSize: '16px', fontWeight: 'bold', textAlign: 'center', margin: '-4px 0 0 0' })
  ],
  style: {
    position: 'top-right', padding: '4px 10px',
    backgroundColor: 'white', border: '2px solid #333'
  }
});
Map.add(northArrow);

// ============================================================
// SECTION 15: MODEL SELECTOR UI
// ============================================================

var modelNames = MODELS.map(function (m) { return m.name; });
var mapTypes = ['LULC 2018', 'LULC 2024', 'Forest 2018', 'Forest 2024', 'Deforestation'];

var controlPanel = ui.Panel({
  style: { position: 'top-left', padding: '10px 15px', width: '320px' }
});

controlPanel.add(ui.Label({
  value: '🗺️ Karnataka LULC — Model Comparison',
  style: { fontWeight: 'bold', fontSize: '16px', margin: '0 0 10px 0' }
}));

controlPanel.add(ui.Label({
  value: 'Select model and map type to display:',
  style: { fontSize: '12px', color: '#555', margin: '0 0 6px 0' }
}));

var modelSelect = ui.Select({
  items: modelNames, value: modelNames[0],
  style: { stretch: 'horizontal', margin: '0 0 6px 0' },
  onChange: updateMap
});

var mapTypeSelect = ui.Select({
  items: mapTypes, value: mapTypes[0],
  style: { stretch: 'horizontal', margin: '0 0 10px 0' },
  onChange: updateMap
});

controlPanel.add(ui.Label('Model:', { fontWeight: 'bold', fontSize: '13px' }));
controlPanel.add(modelSelect);
controlPanel.add(ui.Label('Map Type:', { fontWeight: 'bold', fontSize: '13px' }));
controlPanel.add(mapTypeSelect);

// Download link label
var downloadLabel = ui.Label({
  value: 'Click "Get PNG Link" below',
  style: { fontSize: '11px', color: '#666', margin: '0 0 4px 0', whiteSpace: 'pre-wrap' }
});
controlPanel.add(downloadLabel);

// Get PNG button
var pngButton = ui.Button({
  label: '📥 Get PNG Download Link',
  style: { stretch: 'horizontal', margin: '0 0 6px 0' },
  onClick: function () {
    var mName = modelSelect.getValue();
    var mType = mapTypeSelect.getValue();
    var safe = '';
    MODELS.forEach(function (m) { if (m.name === mName) safe = m.safe; });
    var r = getModelResults(safe);
    if (!r) { downloadLabel.setValue('❌ Model not found'); return; }
    var img, vis, desc;
    if (mType === 'LULC 2018') { img = r.lulc2018; vis = lulcVis; desc = 'LULC_2018'; }
    else if (mType === 'LULC 2024') { img = r.lulc2024; vis = lulcVis; desc = 'LULC_2024'; }
    else if (mType === 'Forest 2018') { img = r.forest2018; vis = forestVis; desc = 'Forest_2018'; }
    else if (mType === 'Forest 2024') { img = r.forest2024; vis = forestVis; desc = 'Forest_2024'; }
    else { img = r.deforestation; vis = deforestVis; desc = 'Deforestation'; }

    downloadLabel.setValue('⏳ Generating PNG link... please wait');
    var bbox = kGeo.bounds();
    var visImg = img.visualize(vis).unmask(0).clip(kGeo);
    var url = visImg.getThumbURL({
      dimensions: THUMB_DIM, region: bbox, format: 'png'
    });
    downloadLabel.setValue('');
    downloadLabel.setUrl(url);
    downloadLabel.setValue('📥 Click to download: ' + safe + '_' + desc + '.png');
    print('Download URL for ' + mName + ' ' + mType + ':', url);
  }
});
controlPanel.add(pngButton);

// Separator
controlPanel.add(ui.Label({
  value: '──────────────────────',
  style: { color: '#ccc', margin: '6px 0' }
}));

// Area stats button
var statsLabel = ui.Label({ value: '', style: { fontSize: '11px', whiteSpace: 'pre-wrap' } });
controlPanel.add(ui.Button({
  label: '📊 Compute Area Statistics',
  style: { stretch: 'horizontal', margin: '0 0 6px 0' },
  onClick: function () {
    var mName = modelSelect.getValue();
    var mType = mapTypeSelect.getValue();
    var safe = '';
    MODELS.forEach(function (m) { if (m.name === mName) safe = m.safe; });
    var r = getModelResults(safe);
    if (!r) { statsLabel.setValue('❌ Model not found'); return; }
    var img = (mType.indexOf('2018') >= 0 || mType === 'Deforestation') ? r.lulc2018 : r.lulc2024;
    var yearLabel = (mType.indexOf('2024') >= 0) ? '2024' : '2018';

    statsLabel.setValue('⏳ Computing area stats...');

    var areaImg = ee.Image.pixelArea().divide(1e6); // km²
    var classAreas = img.addBands(areaImg).reduceRegion({
      reducer: ee.Reducer.sum().group({ groupField: 0, groupName: 'class' }),
      geometry: kGeo, scale: 500, maxPixels: 1e13, tileScale: 8
    });

    classAreas.evaluate(function (result) {
      var txt = mName + ' — ' + yearLabel + ':\n';
      if (result && result.groups) {
        result.groups.forEach(function (g) {
          var cn = (g['class'] < CLASS_NAMES.length) ? CLASS_NAMES[g['class']] : 'Unknown';
          txt += '  ' + cn + ': ' + (g.sum || 0).toFixed(1) + ' km²\n';
        });
      }
      statsLabel.setValue(txt);
    });
  }
}));
controlPanel.add(statsLabel);

Map.add(controlPanel);

// ============================================================
// SECTION 16: MAP UPDATE FUNCTION
// ============================================================

function updateMap() {
  // Remove existing classification layers
  while (Map.layers().length() > 0) {
    Map.layers().remove(Map.layers().get(0));
  }

  var mName = modelSelect.getValue();
  var mType = mapTypeSelect.getValue();
  var safe = '';
  MODELS.forEach(function (m) { if (m.name === mName) safe = m.safe; });

  // Lazy computation — only trains model when first selected
  var r = getModelResults(safe);
  if (!r) return;

  var layerName = mName + ' — ' + mType;

  if (mType === 'LULC 2018') {
    Map.addLayer(r.lulc2018, lulcVis, layerName, true);
  } else if (mType === 'LULC 2024') {
    Map.addLayer(r.lulc2024, lulcVis, layerName, true);
  } else if (mType === 'Forest 2018') {
    Map.addLayer(r.forest2018, forestVis, layerName, true);
  } else if (mType === 'Forest 2024') {
    Map.addLayer(r.forest2024, forestVis, layerName, true);
  } else {
    Map.addLayer(r.deforestation, deforestVis, layerName, true);
  }
}

// Initialize with first model
updateMap();

// ============================================================
// SECTION 17: EXPORT ALL 50 MAPS TO GOOGLE DRIVE
// ============================================================

print('');
print('=== EXPORT TASKS ===');
print('Go to Tasks tab and run all 50 export tasks.');
print('All maps will be saved to Google Drive/' + EXPORT_FOLDER + '/');
print('');

MODELS.forEach(function (m) {
  var r = getModelResults(m.safe);
  var maps = [
    { img: r.lulc2018, vis: lulcVis, suffix: 'LULC_2018' },
    { img: r.lulc2024, vis: lulcVis, suffix: 'LULC_2024' },
    { img: r.forest2018, vis: forestVis, suffix: 'Forest_2018' },
    { img: r.forest2024, vis: forestVis, suffix: 'Forest_2024' },
    { img: r.deforestation, vis: deforestVis, suffix: 'Deforestation_2018_2024' }
  ];

  maps.forEach(function (mapDef) {
    var desc = m.safe + '_' + mapDef.suffix;
    Export.image.toDrive({
      image: mapDef.img.visualize(mapDef.vis).unmask(0).clip(kGeo),
      description: desc,
      folder: EXPORT_FOLDER,
      region: kGeo,
      scale: EXPORT_SCALE,
      maxPixels: 1e13,
      fileFormat: 'GeoTIFF'
    });
  });
});

// ============================================================
// SECTION 18: BATCH PNG DOWNLOAD LINKS
// ============================================================

print('');
print('=== PNG THUMBNAIL LINKS ===');
print('Use the Model Selector panel to get individual PNG links,');
print('or run this section to print all 50 links at once:');
print('');

// Uncomment below to print all 50 PNG URLs (may be slow):
/*
MODELS.forEach(function(m) {
  var r = results[m.safe];
  var maps = [
    {img: r.lulc2018, vis: lulcVis, suffix: 'LULC_2018'},
    {img: r.lulc2024, vis: lulcVis, suffix: 'LULC_2024'},
    {img: r.forest2018, vis: forestVis, suffix: 'Forest_2018'},
    {img: r.forest2024, vis: forestVis, suffix: 'Forest_2024'},
    {img: r.deforestation, vis: deforestVis, suffix: 'Deforestation_2018_2024'}
  ];
  maps.forEach(function(mapDef) {
    var visImg = mapDef.img.visualize(mapDef.vis).unmask(0).clip(kGeo);
    var url = visImg.getThumbURL({
      dimensions: THUMB_DIM, region: kGeo.bounds(), format: 'png'
    });
    print('📥 ' + m.name + ' — ' + mapDef.suffix + ':', url);
  });
});
*/

print('');
print('=== INSTRUCTIONS ===');
print('1. Use the MODEL SELECTOR (top-left panel) to view each model');
print('2. Click "Get PNG Download Link" to download the current map as PNG');
print('3. Go to TASKS tab to export all 50 maps to Google Drive');
print('4. The Code Editor view includes legend + north arrow for screenshots');
print('5. Scale bar is shown automatically at the bottom of the map');
print('');
print('=== MODEL NAME MAPPING ===');
print('  TriSAFNet (Proposed)  → smileRandomForest(500)');
print('  Lightweight CNN-RF  → smileRandomForest(150, maxNodes=50)');
print('  CNN-Only            → smileCart');
print('  CNN-SVM             → libsvm(RBF, cost=10)');
print('  CNN-ExtraTrees      → smileRandomForest(200, bagFraction=0.8)');
print('  CNN-XGBoost         → smileGradientTreeBoost(200, shrinkage=0.05)');
print('  CNN-LightGBM        → smileGradientTreeBoost(100, shrinkage=0.1)');
print('  EfficientNet-B0     → smileRandomForest(300, maxNodes=100)');
print('  RF-Only             → smileRandomForest(200)');
print('  MobileNetV2         → smileNaiveBayes');
