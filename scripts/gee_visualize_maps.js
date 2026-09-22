/*
=============================================================================
  GEE CODE EDITOR: Visualize LULC Classification Maps
  
  After running Colab classification and uploading classified GeoTIFFs
  as GEE Assets, use this script to create publication maps like the
  "Babi" reference maps.
  
  For EACH model, upload these assets:
    - classified_2018 (merged GeoTIFF from Colab)
    - classified_2024 (merged GeoTIFF from Colab)
    
  This script produces 5 map layers per model:
    1. LULC 2018 (all classes)
    2. LULC 2024 (all classes)
    3. Forest only 2018
    4. Forest only 2024
    5. Deforestation 2018→2024
=============================================================================
*/

// ============================================================
// CONFIGURATION — Update these asset paths after upload
// ============================================================

// Study area (Bengaluru region — same as reference maps)
var studyArea = ee.Geometry.Rectangle([77.0, 12.6, 78.0, 13.5]);

// --- MODEL ASSETS ---
// Replace these with your actual asset paths after upload.
// Upload format: earthengine upload image --asset_id users/<username>/CNN_RF/<name> <file.tif>
//
// Example for TriSAFNet:
var trisafnet_2018 = ee.Image('users/YOUR_USERNAME/CNN_RF/TriSAFNet_classified_2018');
var trisafnet_2024 = ee.Image('users/YOUR_USERNAME/CNN_RF/TriSAFNet_classified_2024');

// Example for CNN Only (uncomment and set after upload):
// var cnn_only_2018 = ee.Image('users/YOUR_USERNAME/CNN_RF/CNN_Only_classified_2018');
// var cnn_only_2024 = ee.Image('users/YOUR_USERNAME/CNN_RF/CNN_Only_classified_2024');

// var effnet_2018 = ee.Image('users/YOUR_USERNAME/CNN_RF/EfficientNet_classified_2018');
// var effnet_2024 = ee.Image('users/YOUR_USERNAME/CNN_RF/EfficientNet_classified_2024');

// var cnnrf_2018 = ee.Image('users/YOUR_USERNAME/CNN_RF/CNN_RF_classified_2018');
// var cnnrf_2024 = ee.Image('users/YOUR_USERNAME/CNN_RF/CNN_RF_classified_2024');


// ============================================================
// CLASS DEFINITIONS
// ============================================================
// Class 0: Forest
// Class 1: Plantation
// Class 2: Fallow Land
// Class 3: Water Bodies
// Class 4: Built-up Area

var classNames = ['Forest', 'Plantation', 'Fallow Land', 'Water Bodies', 'Built-up Area'];
var classColors = ['#00770c', '#ba02d6', '#815756', '#0821ff', '#ff0000'];
var deforestColor = '#FFFF00';  // Yellow for deforestation (matching reference)
var forestColor = '#00FF00';    // Green for forest-only maps


// ============================================================
// VISUALIZATION PARAMETERS
// ============================================================

var lulcVis = {
  min: 0,
  max: 4,
  palette: classColors
};

var forestVis = {
  min: 0,
  max: 1,
  palette: ['00000000', forestColor]  // transparent + green
};

var deforestVis = {
  min: 0,
  max: 1,
  palette: ['00000000', deforestColor]  // transparent + yellow
};


// ============================================================
// HELPER FUNCTIONS
// ============================================================

/**
 * Create all 5 map layers for one model.
 * @param {ee.Image} classified2018 - Classified image for 2018
 * @param {ee.Image} classified2024 - Classified image for 2024
 * @param {string} modelName - Display name for the model
 * @param {boolean} show - Whether to show layers by default
 */
function addModelLayers(classified2018, classified2024, modelName, show) {
  // Mask nodata (-1)
  var c2018 = classified2018.updateMask(classified2018.gte(0));
  var c2024 = classified2024.updateMask(classified2024.gte(0));
  
  // 1. LULC 2018 (all classes)
  Map.addLayer(c2018, lulcVis, modelName + ' — LULC 2018', show);
  
  // 2. LULC 2024 (all classes)
  Map.addLayer(c2024, lulcVis, modelName + ' — LULC 2024', false);
  
  // 3. Forest only 2018 (class 0)
  var forest2018 = c2018.eq(0).selfMask();
  Map.addLayer(forest2018, {palette: [forestColor]}, 
               modelName + ' — Forest 2018', false);
  
  // 4. Forest only 2024 (class 0)
  var forest2024 = c2024.eq(0).selfMask();
  Map.addLayer(forest2024, {palette: [forestColor]}, 
               modelName + ' — Forest 2024', false);
  
  // 5. Deforestation 2018→2024 (forest in 2018 AND not forest in 2024)
  var deforestation = c2018.eq(0).and(c2024.neq(0)).selfMask();
  Map.addLayer(deforestation, {palette: [deforestColor]}, 
               modelName + ' — Deforestation 2018-2024', false);
  
  return {
    lulc2018: c2018,
    lulc2024: c2024,
    forest2018: forest2018,
    forest2024: forest2024,
    deforestation: deforestation
  };
}


/**
 * Compute and print area statistics for one model.
 */
function computeStats(classified, year, modelName) {
  var stats = {};
  for (var i = 0; i < classNames.length; i++) {
    var classMask = classified.eq(i);
    var area = classMask.multiply(ee.Image.pixelArea()).divide(1e6); // km²
    var totalArea = area.reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: studyArea,
      scale: 10,
      maxPixels: 1e13
    });
    stats[classNames[i]] = totalArea;
  }
  print(modelName + ' — ' + year + ' Area Statistics:', stats);
}


// ============================================================
// ADD LEGEND
// ============================================================

function addLegend() {
  var legend = ui.Panel({
    style: {
      position: 'bottom-left',
      padding: '8px 15px'
    }
  });
  
  legend.add(ui.Label({
    value: 'Land Cover Classification',
    style: {fontWeight: 'bold', fontSize: '14px', margin: '0 0 4px 0'}
  }));
  
  // LULC classes
  for (var i = 0; i < classNames.length; i++) {
    var row = ui.Panel({
      widgets: [
        ui.Label({
          style: {
            backgroundColor: classColors[i],
            padding: '8px', margin: '0 4px 4px 0',
            border: '1px solid #999'
          }
        }),
        ui.Label({value: classNames[i], style: {margin: '0 0 4px 0'}})
      ],
      layout: ui.Panel.Layout.Flow('horizontal')
    });
    legend.add(row);
  }
  
  // Deforestation entry
  var defRow = ui.Panel({
    widgets: [
      ui.Label({
        style: {
          backgroundColor: deforestColor,
          padding: '8px', margin: '0 4px 4px 0',
          border: '1px solid #999'
        }
      }),
      ui.Label({value: 'Deforestation (2018-2024)', style: {margin: '0 0 4px 0'}})
    ],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
  legend.add(defRow);
  
  Map.add(legend);
  
  // --- North Arrow ---
  var northArrow = ui.Panel({
    widgets: [
      ui.Label('▲', {fontSize: '20px', fontWeight: 'bold', 
                      textAlign: 'center', margin: '0'}),
      ui.Label('N', {fontSize: '14px', fontWeight: 'bold', 
                      textAlign: 'center', margin: '-4px 0 0 0'})
    ],
    style: {
      position: 'top-right',
      padding: '4px 8px',
      backgroundColor: 'white',
      border: '1px solid #999'
    }
  });
  Map.add(northArrow);
  
  // Note: GEE Code Editor shows a scale bar automatically at bottom-right
}


// ============================================================
// MAIN EXECUTION
// ============================================================

// Center map on study area
Map.centerObject(studyArea, 10);
Map.setOptions('SATELLITE');

// Add legend
addLegend();

// --- Add TriSAFNet layers (shown by default) ---
var trisafnet = addModelLayers(trisafnet_2018, trisafnet_2024, 'TriSAFNet (Proposed)', true);

// --- Add baseline models (hidden by default, toggle in Layers panel) ---
// Uncomment after uploading the corresponding assets:

// var cnnOnly = addModelLayers(cnn_only_2018, cnn_only_2024, 'CNN Only', false);
// var effnet = addModelLayers(effnet_2018, effnet_2024, 'EfficientNet-B0', false);
// var cnnrf = addModelLayers(cnnrf_2018, cnnrf_2024, 'Lightweight CNN-RF', false);

// --- Print area statistics ---
computeStats(trisafnet.lulc2018, '2018', 'TriSAFNet');
computeStats(trisafnet.lulc2024, '2024', 'TriSAFNet');


// ============================================================
// EXPORT MAPS AS HIGH-RES IMAGES (optional)
// ============================================================
// Uncomment to export publication-quality maps to Google Drive

/*
// Export LULC 2018
Export.map.toDrive({
  description: 'TriSAFNet_LULC_2018',
  region: studyArea,
  scale: 10,
  maxPixels: 1e13
});

// Export Forest 2018  
Export.image.toDrive({
  image: trisafnet.forest2018.visualize({palette: [forestColor]}),
  description: 'TriSAFNet_Forest_2018',
  region: studyArea,
  scale: 10,
  maxPixels: 1e13
});

// Export Deforestation
Export.image.toDrive({
  image: trisafnet.deforestation.visualize({palette: [deforestColor]}),
  description: 'TriSAFNet_Deforestation_2018_2024',
  region: studyArea,
  scale: 10,
  maxPixels: 1e13
});
*/


// ============================================================
// INSTRUCTIONS
// ============================================================
print('=== USAGE ===');
print('1. Toggle layers in the Layers panel (top-right) to switch between maps');
print('2. Each model has 5 layers: LULC 2018, LULC 2024, Forest 2018, Forest 2024, Deforestation');
print('3. Take screenshots for your paper using the GEE map view');
print('4. The satellite basemap + place names appear automatically');
print('');
print('=== TO UPLOAD ASSETS ===');
print('In terminal (after installing earthengine CLI):');
print('  earthengine upload image --asset_id users/USERNAME/CNN_RF/TriSAFNet_classified_2018 path/to/classified.tif');
