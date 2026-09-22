# ============================================================
# configs/config.py — All hyperparameters for the LULC pipeline
# ============================================================

RANDOM_SEED = 42

# ----- DATA -----
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"
RUNS_DIR = "./runs"
PATCH_SIZE = 21
PATCH_SIZES = [7, 15, 21]          # for spatial context ablation study
N_BANDS = 27
N_CLASSES = 5
N_FOLDS = 5
SPATIAL_CV = True        # True = spatial blocks, False = stratified k-fold  [revision: spatial-block CV]
TEST_RATIO = 0.2

BAND_NAMES = [
    # SAR C-band (0-2)
    "VV", "VH", "RVI",
    # SAR L-band (3-4)
    "HH", "HV",
    # S2 annual reflectance (5-16)
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B11", "B12",
    # S2 annual spectral indices (17-23)
    "NDVI", "NDWI", "MNDWI", "EVI", "BSI", "SAVI", "NDBI",
    # Thermal (24)
    "LST",
    # Topographic (25-26)
    "elevation", "slope"
]

# Band index ranges for multi-branch models (must be contiguous slices)
SAR_BANDS = list(range(0, 5))       # VV, VH, RVI, HH, HV
OPTICAL_BANDS = list(range(5, 25))  # S2 annual + indices + LST
TOPO_BANDS = list(range(25, 27))    # elevation, slope

CLASS_NAMES = ["Forest", "Plantation", "Fallow Land", "Water Bodies", "Built-up Area"]
CLASS_COLORS = ["#00770c", "#ba02d6", "#815756", "#0821ff", "#ff0000"]

# ----- AUGMENTATION -----
AUGMENT_GEOMETRIC = True
AUGMENT_PHYSICS = True
AUGMENT_NOISE = True               # GaussianNoise layers inside CNN models (training-time only)
GAUSSIAN_NOISE_STDDEV = 0.02       # std for GaussianNoise layers in models.py

# ----- PATHS -----
RASTER_2018 = "./data/stack_2018.tif"
RASTER_2024 = "./data/stack_2024.tif"
POINTS_2018_CSV = "./data/training_points_2018.csv"
POINTS_2024_CSV = "./data/training_points_2024.csv"
SAMPLES_2018_CSV = "./data/training_samples_2018.csv"
SAMPLES_2024_CSV = "./data/training_samples_2024.csv"
EXTRA_POINTS_2018_CSV = "./data/extra_training_points_2018.csv"
EXTRA_POINTS_2024_CSV = "./data/extra_training_points_2024.csv"
N_SAMPLES_PER_CLASS = 500
N_EXTRA_PER_CLASS = 1200       # max extra Dynamic World points per class

# ============================================================
# MODEL 1 (PROPOSED): TriSAFNet
# ============================================================
TRISAF_SAR_FILTERS = [32, 64]
TRISAF_OPTICAL_FILTERS = [64, 128, 128]
TRISAF_TOPO_FILTERS = [8, 16]
TRISAF_SE_REDUCTION = 4          # SE block squeeze ratio
TRISAF_CBAM_REDUCTION = 8        # CBAM channel attention squeeze ratio
TRISAF_CBAM_KERNEL = 7           # CBAM spatial attention conv kernel
TRISAF_DENSE = 64                # dense layer before softmax
TRISAF_DROPOUT = 0.3
TRISAF_L2 = 0.005
TRISAF_LR = 0.001
TRISAF_EPOCHS = 100
TRISAF_BATCH_SIZE = 64
TRISAF_LABEL_SMOOTHING = 0.05
TRISAF_CLASS_WEIGHTS = True

# ============================================================
# MODEL 2: Lightweight CNN-RF (previous proposed, now baseline)
# ============================================================
LW_CNN_FILTERS = [16, 32, 64]
LW_CNN_KERNEL = 3
LW_CNN_DROPOUT = 0.4
LW_CNN_L2 = 0.005
LW_CNN_LR = 0.001
LW_CNN_EPOCHS = 50
LW_CNN_BATCH_SIZE = 64
LW_CNN_LABEL_SMOOTHING = 0.1
LW_CNN_CLASS_WEIGHTS = True

# ============================================================
# MODEL 3: Traditional CNN-RF
# ============================================================
TRAD_CNN_FILTERS = [32, 64, 128]
TRAD_CNN_DENSE = [256, 128]
TRAD_CNN_EPOCHS = 50

# ============================================================
# MODEL 4: CNN-Only (end-to-end DL baseline)
# ============================================================
CNN_ONLY_FILTERS = [32, 64, 128]
CNN_ONLY_DENSE = [256, 128]
CNN_ONLY_EPOCHS = 50

# ============================================================
# MODEL 5: CNN-SVM
# ============================================================
SVM_C = 10.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"

# ============================================================
# MODEL 6: CNN-ExtraTrees
# ============================================================
ET_N_ESTIMATORS = 200
ET_MAX_DEPTH = None
ET_MIN_SAMPLES_SPLIT = 5

# ============================================================
# MODEL 7: CNN-KNN
# ============================================================
KNN_N_NEIGHBORS = 7
KNN_WEIGHTS = "distance"
KNN_METRIC = "minkowski"

# ============================================================
# MODEL 8: CNN-XGBoost
# ============================================================
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 5
XGB_LR = 0.05

# ============================================================
# MODEL 9: CNN-LightGBM
# ============================================================
LGBM_N_ESTIMATORS = 200
LGBM_MAX_DEPTH = 5
LGBM_LR = 0.05

# ============================================================
# MODEL 10: MobileNetV2 (DL baseline)
# ============================================================
MOBILENET_EPOCHS = 50
MOBILENET_BATCH_SIZE = 32
MOBILENET_LR = 0.0003

# ============================================================
# MODEL 11: RF-Only (no CNN)
# ============================================================
RF_N_ESTIMATORS = 150
RF_MAX_DEPTH = 15
RF_CRITERION = "gini"
RF_ONLY_ESTIMATORS = 200
RF_ONLY_MAX_DEPTH = None

# ============================================================
# MODEL 12: SVM-Only (no CNN)
# ============================================================
SVM_ONLY_C = 10.0
SVM_ONLY_KERNEL = "rbf"
SVM_ONLY_GAMMA = "scale"

# ============================================================
# MODEL 13: XGBoost-Only (no CNN)
# ============================================================
XGB_ONLY_N_ESTIMATORS = 300
XGB_ONLY_MAX_DEPTH = 6
XGB_ONLY_LR = 0.05

# ============================================================
# MODEL 14: ANN (MLP baseline)
# ============================================================
ANN_HIDDEN = (256, 128, 64)
ANN_MAX_ITER = 200

# ============================================================
# MODEL 15: EfficientNet-B0 (custom MBConv for small patches)
# ============================================================
EFFNET_FILTERS = [16, 24, 40, 80]
EFFNET_EXPAND_RATIO = 4
EFFNET_EPOCHS = 60
EFFNET_BATCH_SIZE = 64
EFFNET_LR = 0.001

# ============================================================
# MODEL 16: Compact Vision Transformer (ViT-Tiny)
# ============================================================
VIT_PATCH_TOKEN = 3       # sub-patch size within the 21x21 patch
VIT_EMBED_DIM = 64
VIT_NUM_HEADS = 4
VIT_NUM_LAYERS = 4
VIT_MLP_DIM = 128
VIT_DROPOUT = 0.3
VIT_EPOCHS = 60
VIT_BATCH_SIZE = 64
VIT_LR = 0.001

# ============================================================
# PHYSICS-INFORMED AUGMENTATION
# ============================================================
SPECKLE_SIGMA = 0.15      # multiplicative speckle std for SAR
ATMO_BRIGHTNESS = 0.05    # additive brightness shift for optical
ATMO_HAZE = 0.03          # additive haze noise for optical
PHYSICS_AUG_FRACTION = 0.6  # fraction of samples to augment (rest kept clean)

# ============================================================
# BENCHMARKING
# ============================================================
BENCHMARK_N_RUNS = 5


