"""
models.py
All 14 model architectures for the LULC comparative study.

Models follow a standard interface:
    __init__(input_shape)  – build/compile
    fit(X_train, y_train, X_val=None, y_val=None)
    predict(X)             – class labels
    predict_proba(X)       – (N, N_CLASSES) probabilities
    get_param_count()      – int
    get_name()             – string
"""

import os, sys
_dll_dir = os.path.join(sys.prefix, 'Library', 'bin')
if os.path.isdir(_dll_dir):
    os.environ['PATH'] = _dll_dir + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(_dll_dir)

# Point XLA to the CUDA data dir so it can find libdevice.10.bc
_cuda_data_dir = os.path.join(sys.prefix, 'Library')
if os.path.isfile(os.path.join(_cuda_data_dir, 'bin', 'libdevice.10.bc')):
    os.environ.setdefault('XLA_FLAGS', f'--xla_gpu_cuda_data_dir={_cuda_data_dir}')

import numpy as np
import tensorflow as tf
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import *

# Use legacy Adam optimizer (experimental AdamW forces XLA which needs ptxas.exe)
# L2 regularization is already applied in all models, so Adam is functionally equivalent.
from tensorflow.keras.optimizers import Adam as AdamW


# ============================================================
# UTILITIES
# ============================================================

import logging
_model_logger = logging.getLogger('models')


class TrainingLogger(callbacks.Callback):
    """Logs callback actions (LR reduction, early stop, checkpoint) via Python logging."""

    def __init__(self, model_name=''):
        super().__init__()
        self._name = model_name
        self._prev_lr = None
        self._total_epochs = 0
        self._best_val_loss = float('inf')
        self._best_epoch = 0

    def on_train_begin(self, logs=None):
        self._prev_lr = float(self.model.optimizer.learning_rate)
        self._total_epochs = self.params.get('epochs', 0)
        _model_logger.info(f"  [{self._name}] Training started "
                           f"(lr={self._prev_lr:.2e}, max_epochs={self._total_epochs})")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        cur_lr = float(self.model.optimizer.learning_rate)
        if self._prev_lr is not None and cur_lr < self._prev_lr:
            _model_logger.info(
                f"  [{self._name}] ReduceLROnPlateau: lr {self._prev_lr:.2e} -> {cur_lr:.2e} "
                f"(epoch {epoch+1})"
            )
        self._prev_lr = cur_lr

        val_loss = logs.get('val_loss', logs.get('loss', None))
        if val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_epoch = epoch + 1

    def on_train_end(self, logs=None):
        actual_epochs = len(self.model.history.epoch) if hasattr(self.model, 'history') else 0
        if self.model.stop_training and actual_epochs < self._total_epochs:
            _model_logger.info(
                f"  [{self._name}] EarlyStopping at epoch {actual_epochs}/{self._total_epochs}, "
                f"best weights restored from epoch {self._best_epoch} "
                f"(val_loss={self._best_val_loss:.5f})"
            )
        else:
            _model_logger.info(
                f"  [{self._name}] Training completed all {self._total_epochs} epochs, "
                f"best epoch was {self._best_epoch} (val_loss={self._best_val_loss:.5f})"
            )


class _CheckpointLogger(callbacks.ModelCheckpoint):
    """ModelCheckpoint that also logs when a checkpoint is saved."""

    def __init__(self, *args, model_name='', **kwargs):
        super().__init__(*args, **kwargs)
        self._name = model_name

    def on_epoch_end(self, epoch, logs=None):
        prev_best = self.best
        super().on_epoch_end(epoch, logs)
        if self.best != prev_best:
            _model_logger.info(
                f"  [{self._name}] ModelCheckpoint: saved best weights "
                f"(epoch {epoch+1}, {self.monitor}={self.best:.5f})"
            )


def get_class_weights(y):
    """Compute balanced class weights."""
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes.astype(int), weights))


def se_block(x, reduction=4):
    """Squeeze-and-Excitation channel attention."""
    channels = x.shape[-1]
    squeeze = layers.GlobalAveragePooling2D()(x)
    excite = layers.Dense(channels // reduction, activation='relu')(squeeze)
    excite = layers.Dense(channels, activation='sigmoid')(excite)
    excite = layers.Reshape((1, 1, channels))(excite)
    return layers.Multiply()([x, excite])


def cbam_block(x, reduction=8, kernel_size=7):
    """Convolutional Block Attention Module (channel + spatial)."""
    channels = x.shape[-1]

    # --- Channel Attention ---
    avg = layers.GlobalAveragePooling2D()(x)
    mx = layers.GlobalMaxPooling2D()(x)
    shared_d1 = layers.Dense(channels // reduction, activation='relu')
    shared_d2 = layers.Dense(channels, activation='linear')
    avg_out = shared_d2(shared_d1(avg))
    mx_out = shared_d2(shared_d1(mx))
    channel_att = layers.Activation('sigmoid')(layers.Add()([avg_out, mx_out]))
    channel_att = layers.Reshape((1, 1, channels))(channel_att)
    x = layers.Multiply()([x, channel_att])

    # --- Spatial Attention ---
    avg_spatial = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_spatial = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate()([avg_spatial, max_spatial])
    spatial_att = layers.Conv2D(1, kernel_size, padding='same', activation='sigmoid')(concat)
    x = layers.Multiply()([x, spatial_att])
    return x


def cross_source_attention(f_a, f_b, f_c, name_prefix='csa'):
    """Cross-Source Attention: each branch is gated by context from the other two.

    Lightweight alternative to full cross-attention: uses global-pool of
    sister branches to produce a channel-wise sigmoid gate, enabling each
    source to be modulated by information from the other sensor modalities.
    """
    g_a = layers.GlobalAveragePooling2D()(f_a)
    g_b = layers.GlobalAveragePooling2D()(f_b)
    g_c = layers.GlobalAveragePooling2D()(f_c)

    ch_a, ch_b, ch_c = f_a.shape[-1], f_b.shape[-1], f_c.shape[-1]

    ctx_a = layers.Concatenate()([g_b, g_c])
    gate_a = layers.Dense(ch_a, activation='sigmoid',
                          name=f'{name_prefix}_gate_a')(ctx_a)
    gate_a = layers.Reshape((1, 1, ch_a))(gate_a)
    f_a = layers.Multiply(name=f'{name_prefix}_mul_a')([f_a, gate_a])

    ctx_b = layers.Concatenate()([g_a, g_c])
    gate_b = layers.Dense(ch_b, activation='sigmoid',
                          name=f'{name_prefix}_gate_b')(ctx_b)
    gate_b = layers.Reshape((1, 1, ch_b))(gate_b)
    f_b = layers.Multiply(name=f'{name_prefix}_mul_b')([f_b, gate_b])

    ctx_c = layers.Concatenate()([g_a, g_b])
    gate_c = layers.Dense(ch_c, activation='sigmoid',
                          name=f'{name_prefix}_gate_c')(ctx_c)
    gate_c = layers.Reshape((1, 1, ch_c))(gate_c)
    f_c = layers.Multiply(name=f'{name_prefix}_mul_c')([f_c, gate_c])

    return f_a, f_b, f_c


def _pretrain_cnn(cnn, X_train, y_train, X_val, y_val,
                  lr=LW_CNN_LR, epochs=LW_CNN_EPOCHS,
                  batch_size=LW_CNN_BATCH_SIZE,
                  label_smoothing=LW_CNN_LABEL_SMOOTHING,
                  use_class_weights=LW_CNN_CLASS_WEIGHTS,
                  patience=8, model_name='CNN-ML'):
    """Shared CNN pre-training routine for all CNN+ML hybrids.

    Adds a temp softmax head, trains, strips it, returns keras History.
    """
    out = layers.Dense(N_CLASSES, activation='softmax')(cnn.output)
    trainer = models.Model(cnn.input, out)

    cw = get_class_weights(y_train) if use_class_weights else None

    # Label smoothing
    if label_smoothing > 0:
        y_smooth = tf.one_hot(y_train.astype(int), N_CLASSES)
        y_smooth = y_smooth * (1 - label_smoothing) + label_smoothing / N_CLASSES
        trainer.compile(optimizer=AdamW(learning_rate=lr),
                        loss='categorical_crossentropy', metrics=['accuracy'])
        train_y = y_smooth
        if X_val is not None:
            val_data = (X_val, tf.one_hot(y_val.astype(int), N_CLASSES))
        else:
            val_data = None
    else:
        trainer.compile(optimizer=AdamW(learning_rate=lr),
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        train_y = y_train
        val_data = (X_val, y_val) if X_val is not None else None

    mon = 'val_loss' if val_data else 'loss'
    history = trainer.fit(
        X_train, train_y, validation_data=val_data,
        epochs=epochs, batch_size=batch_size,
        callbacks=[
            callbacks.ReduceLROnPlateau(monitor=mon, factor=0.5, patience=5, min_lr=1e-5),
            callbacks.EarlyStopping(monitor=mon, patience=patience, restore_best_weights=True),
            TrainingLogger(model_name),
        ],
        class_weight=cw, verbose=1
    )
    return history


def build_lightweight_cnn_feature_extractor(input_shape):
    """Shared lightweight CNN backbone used by all CNN+ML hybrids."""
    inp = layers.Input(shape=input_shape)
    x = inp
    if AUGMENT_NOISE:
        x = layers.GaussianNoise(GAUSSIAN_NOISE_STDDEV)(x)

    for i, f in enumerate(LW_CNN_FILTERS):
        x = layers.Conv2D(f, LW_CNN_KERNEL, activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(LW_CNN_L2),
                          name=f'conv_{i+1}')(x)
        x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
        if x.shape[1] is not None and x.shape[1] >= 4:
            x = layers.MaxPooling2D((2, 2), name=f'pool_{i+1}')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', name='feature_proj')(x)
    x = layers.Dropout(LW_CNN_DROPOUT)(x)
    return models.Model(inp, x, name='LW_CNN_FeatureExtractor')


# ============================================================
# MODEL 1 (PROPOSED): TriSAFNet — Tri-Source Attention Fusion
# ============================================================

class TriSAFNet:
    """Tri-Source Attention Fusion Network (proposed model).

    Three parallel branches (SAR / Optical / Topo) each with SE attention,
    fused and refined with CBAM, classified via Dense softmax.
    Supports ablation flags to disable individual components.
    """

    def __init__(self, input_shape, use_se=True, use_cbam=True, use_csa=True, single_branch=False):
        self.input_shape = input_shape
        self.use_se = use_se
        self.use_cbam = use_cbam
        self.use_csa = use_csa
        self.single_branch = single_branch
        self._build()

    def _build(self):
        inp = layers.Input(shape=self.input_shape)
        reg = regularizers.l2(TRISAF_L2)

        if self.single_branch:
            # --- Single-branch mode: run one conv pipeline on all bands ---
            x = layers.GaussianNoise(GAUSSIAN_NOISE_STDDEV)(inp) if AUGMENT_NOISE else inp
            for f in [32, 64, 64]:
                x = layers.Conv2D(f, 3, padding='same', activation='relu',
                                  kernel_regularizer=reg)(x)
                x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2), padding='same')(x)

            if self.use_cbam:
                x = cbam_block(x, reduction=TRISAF_CBAM_REDUCTION,
                               kernel_size=TRISAF_CBAM_KERNEL)

            fused = x
        else:
            # --- Split input by source ---
            x_sar  = layers.Lambda(lambda t: t[:, :, :, SAR_BANDS[0]:SAR_BANDS[-1]+1],
                                   name='split_sar')(inp)
            x_opt  = layers.Lambda(lambda t: t[:, :, :, OPTICAL_BANDS[0]:OPTICAL_BANDS[-1]+1],
                                   name='split_opt')(inp)
            x_topo = layers.Lambda(lambda t: t[:, :, :, TOPO_BANDS[0]:TOPO_BANDS[-1]+1],
                                   name='split_topo')(inp)

            # --- SAR Branch (2 conv blocks + SE) ---
            s = layers.GaussianNoise(GAUSSIAN_NOISE_STDDEV)(x_sar) if AUGMENT_NOISE else x_sar
            for f in TRISAF_SAR_FILTERS:
                s = layers.Conv2D(f, 3, padding='same', activation='relu',
                                  kernel_regularizer=reg)(s)
                s = layers.BatchNormalization()(s)
            s = layers.MaxPooling2D((2, 2), padding='same')(s)
            if self.use_se:
                s = se_block(s, reduction=TRISAF_SE_REDUCTION)

            # --- Optical Branch (3 conv blocks + residual + SE) ---
            o = layers.GaussianNoise(GAUSSIAN_NOISE_STDDEV)(x_opt) if AUGMENT_NOISE else x_opt
            o = layers.Conv2D(TRISAF_OPTICAL_FILTERS[0], 3, padding='same', activation='relu',
                              kernel_regularizer=reg)(o)
            o = layers.BatchNormalization()(o)
            o_shortcut = layers.Conv2D(TRISAF_OPTICAL_FILTERS[1], 1, padding='same',
                                       kernel_regularizer=reg)(o)
            o = layers.Conv2D(TRISAF_OPTICAL_FILTERS[1], 3, padding='same', activation='relu',
                              kernel_regularizer=reg)(o)
            o = layers.BatchNormalization()(o)
            o = layers.Add()([o, o_shortcut])
            o = layers.Activation('relu')(o)
            o = layers.Conv2D(TRISAF_OPTICAL_FILTERS[2], 3, padding='same', activation='relu',
                              kernel_regularizer=reg)(o)
            o = layers.BatchNormalization()(o)
            o = layers.MaxPooling2D((2, 2), padding='same')(o)
            if self.use_se:
                o = se_block(o, reduction=TRISAF_SE_REDUCTION)

            # --- Topographic Branch (2 conv blocks + SE) ---
            t = x_topo
            for f in TRISAF_TOPO_FILTERS:
                t = layers.Conv2D(f, 3, padding='same', activation='relu',
                                  kernel_regularizer=reg)(t)
                t = layers.BatchNormalization()(t)
            t = layers.MaxPooling2D((2, 2), padding='same')(t)
            if self.use_se:
                t = se_block(t, reduction=TRISAF_SE_REDUCTION)

            # --- Cross-Source Attention ---
            if self.use_csa:
                s, o, t = cross_source_attention(s, o, t)

            # --- Fusion + CBAM ---
            fused = layers.Concatenate()([s, o, t])
            if self.use_cbam:
                fused = cbam_block(fused, reduction=TRISAF_CBAM_REDUCTION,
                                   kernel_size=TRISAF_CBAM_KERNEL)

        # --- Classification Head ---
        x = layers.GlobalAveragePooling2D()(fused)
        x = layers.Dropout(TRISAF_DROPOUT)(x)
        x = layers.Dense(TRISAF_DENSE, activation='relu')(x)
        out = layers.Dense(N_CLASSES, activation='softmax')(x)

        self.model = models.Model(inp, out, name='TriSAFNet')
        self.model.compile(
            optimizer=AdamW(learning_rate=TRISAF_LR),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        cw = get_class_weights(y_train) if TRISAF_CLASS_WEIGHTS else None

        # Label smoothing
        y_smooth = tf.one_hot(y_train.astype(int), N_CLASSES)
        y_smooth = y_smooth * (1 - TRISAF_LABEL_SMOOTHING) + TRISAF_LABEL_SMOOTHING / N_CLASSES

        if X_val is not None:
            val_data = (X_val, tf.one_hot(y_val.astype(int), N_CLASSES))
        else:
            val_data = None

        mon = 'val_loss' if val_data else 'loss'
        import tempfile, os
        ckpt_path = os.path.join(tempfile.gettempdir(), 'trisafnet_best_weights.h5')
        self.history = self.model.fit(
            X_train, y_smooth, validation_data=val_data,
            epochs=TRISAF_EPOCHS, batch_size=TRISAF_BATCH_SIZE,
            callbacks=[
                _CheckpointLogger(ckpt_path, monitor=mon,
                                  save_best_only=True, save_weights_only=True,
                                  model_name='TriSAFNet'),
                callbacks.ReduceLROnPlateau(monitor=mon, factor=0.5, patience=12, min_lr=1e-5),
                callbacks.EarlyStopping(monitor=mon, patience=20, restore_best_weights=True),
                TrainingLogger('TriSAFNet'),
            ],
            class_weight=cw, verbose=1
        )
        if os.path.exists(ckpt_path):
            self.model.load_weights(ckpt_path)
            os.remove(ckpt_path)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X, batch_size=256), axis=1)

    def predict_proba(self, X):
        return self.model.predict(X, batch_size=256)

    def get_param_count(self):
        return self.model.count_params()

    def get_name(self):
        return "TriSAFNet (Proposed)"

    # --- Extra methods for paper figures ---

    def get_attention_maps(self, X_sample):
        """Extract CBAM spatial attention maps for visualization."""
        # Find the CBAM spatial sigmoid layer
        spatial_sigmoid = None
        for layer in self.model.layers:
            if isinstance(layer, layers.Conv2D) and layer.output_shape[-1] == 1:
                spatial_sigmoid = layer
        if spatial_sigmoid is None:
            print("Could not find CBAM spatial attention layer")
            return None
        sub = models.Model(self.model.input, spatial_sigmoid.output)
        return sub.predict(X_sample, batch_size=64)


# ============================================================
# MODEL 2: Lightweight CNN-RF (hybrid baseline)
# ============================================================

class LightweightCNNRF:
    """Lightweight CNN feature extractor + Random Forest classifier."""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.cnn = build_lightweight_cnn_feature_extractor(input_shape)
        self.rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            criterion=RF_CRITERION, min_samples_split=5, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_SEED, verbose=1
        )
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.history = _pretrain_cnn(self.cnn, X_train, y_train, X_val, y_val,
                                     model_name='Lightweight CNN-RF')
        features = self.cnn.predict(X_train, batch_size=256)
        print(f"  CNN feature vector size: {features.shape[1]}")
        print(f"  Fitting RF on {len(features)} extracted features...")
        self.rf.fit(features, y_train)
        print(f"  RF fitting complete.")
        return self

    def predict(self, X):
        return self.rf.predict(self.cnn.predict(X, batch_size=256))

    def predict_proba(self, X):
        return self.rf.predict_proba(self.cnn.predict(X, batch_size=256))

    def get_feature_importances(self):
        return self.rf.feature_importances_

    def get_param_count(self):
        return self.cnn.count_params()

    def get_name(self):
        return "Lightweight CNN-RF"


# ============================================================
# MODEL 3: Traditional CNN-RF
# ============================================================

class TraditionalCNNRF:
    """Deeper CNN with dense layers + RF (traditional hybrid)."""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self._build()

    def _build(self):
        inp = layers.Input(shape=self.input_shape)
        x = inp
        if AUGMENT_NOISE:
            x = layers.GaussianNoise(GAUSSIAN_NOISE_STDDEV)(x)

        for f in TRAD_CNN_FILTERS:
            x = layers.Conv2D(f, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(0.005))(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Flatten()(x)
        for units in TRAD_CNN_DENSE:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.3)(x)

        self.cnn = models.Model(inp, x, name='Trad_CNN_FeatureExtractor')
        out = layers.Dense(N_CLASSES, activation='softmax')(x)
        self.cnn_trainer = models.Model(inp, out, name='Trad_CNN_Trainer')
        self.cnn_trainer.compile(
            optimizer=AdamW(learning_rate=0.0005),
            loss='categorical_crossentropy', metrics=['accuracy']
        )
        self.rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            n_jobs=-1, random_state=RANDOM_SEED, verbose=1
        )
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        cw = get_class_weights(y_train) if LW_CNN_CLASS_WEIGHTS else None
        y_smooth = tf.one_hot(y_train.astype(int), N_CLASSES)
        y_smooth = y_smooth * (1 - TRISAF_LABEL_SMOOTHING) + TRISAF_LABEL_SMOOTHING / N_CLASSES
        if X_val is not None:
            val_data = (X_val, tf.one_hot(y_val.astype(int), N_CLASSES))
        else:
            val_data = None
        self.history = self.cnn_trainer.fit(
            X_train, y_smooth, validation_data=val_data,
            epochs=TRAD_CNN_EPOCHS, batch_size=64,
            callbacks=[
                callbacks.EarlyStopping(patience=12, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5),
                TrainingLogger('Traditional CNN-RF'),
            ], class_weight=cw, verbose=1
        )
        features = self.cnn.predict(X_train, batch_size=256)
        print(f"  CNN feature vector size: {features.shape[1]}")
        print(f"  Fitting RF on {len(features)} extracted features...")
        self.rf.fit(features, y_train)
        print(f"  RF fitting complete.")
        return self

    def predict(self, X):
        return self.rf.predict(self.cnn.predict(X, batch_size=256))

    def predict_proba(self, X):
        return self.rf.predict_proba(self.cnn.predict(X, batch_size=256))

    def get_param_count(self):
        return self.cnn.count_params()

    def get_name(self):
        return "Traditional CNN-RF"


# ============================================================
# MODEL 4: CNN-Only
# ============================================================

class CNNOnly:
    """End-to-end CNN with softmax (DL baseline)."""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self._build()

    def _build(self):
        inp = layers.Input(shape=self.input_shape)
        x = inp
        if AUGMENT_NOISE:
            x = layers.GaussianNoise(GAUSSIAN_NOISE_STDDEV)(x)

        for f in CNN_ONLY_FILTERS:
            x = layers.Conv2D(f, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(0.005))(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.GlobalAveragePooling2D()(x)
        for units in CNN_ONLY_DENSE:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
        x = layers.Dense(N_CLASSES, activation='softmax')(x)

        self.model = models.Model(inp, x, name='CNN_Only')
        self.model.compile(
            optimizer=AdamW(learning_rate=0.0005),
            loss='categorical_crossentropy', metrics=['accuracy']
        )
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        cw = get_class_weights(y_train) if LW_CNN_CLASS_WEIGHTS else None
        y_smooth = tf.one_hot(y_train.astype(int), N_CLASSES)
        y_smooth = y_smooth * (1 - TRISAF_LABEL_SMOOTHING) + TRISAF_LABEL_SMOOTHING / N_CLASSES
        if X_val is not None:
            val_data = (X_val, tf.one_hot(y_val.astype(int), N_CLASSES))
        else:
            val_data = None
        self.history = self.model.fit(
            X_train, y_smooth, validation_data=val_data,
            epochs=CNN_ONLY_EPOCHS, batch_size=64,
            callbacks=[
                callbacks.EarlyStopping(patience=12, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5),
                TrainingLogger('CNN Only'),
            ], class_weight=cw, verbose=1
        )
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X, batch_size=256), axis=1)

    def predict_proba(self, X):
        return self.model.predict(X, batch_size=256)

    def get_param_count(self):
        return self.model.count_params()

    def get_name(self):
        return "CNN Only"


# ============================================================
# MODEL 5: CNN-SVM
# ============================================================

class CNNSVM:
    """Lightweight CNN features + SVM classifier."""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.cnn = build_lightweight_cnn_feature_extractor(input_shape)
        self.scaler = StandardScaler()
        self.svm = None
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.svm import SVC
        self.history = _pretrain_cnn(self.cnn, X_train, y_train, X_val, y_val,
                                     model_name='CNN-SVM')
        features = self.cnn.predict(X_train, batch_size=256)
        print(f"  CNN feature vector size: {features.shape[1]}")
        features_scaled = self.scaler.fit_transform(features)
        self.svm = SVC(C=SVM_C, kernel=SVM_KERNEL, gamma=SVM_GAMMA,
                       probability=True, random_state=RANDOM_SEED, verbose=True)
        print(f"  Fitting SVM on {len(features_scaled)} extracted features...")
        self.svm.fit(features_scaled, y_train)
        print(f"  SVM fitting complete.")
        return self

    def predict(self, X):
        f = self.scaler.transform(self.cnn.predict(X, batch_size=256))
        return self.svm.predict(f)

    def predict_proba(self, X):
        f = self.scaler.transform(self.cnn.predict(X, batch_size=256))
        return self.svm.predict_proba(f)

    def get_param_count(self):
        return self.cnn.count_params()

    def get_name(self):
        return "CNN-SVM"


# ============================================================
# MODEL 6: CNN-ExtraTrees
# ============================================================

class CNNExtraTrees:
    """Lightweight CNN features + ExtraTreesClassifier."""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.cnn = build_lightweight_cnn_feature_extractor(input_shape)
        self.et = ExtraTreesClassifier(
            n_estimators=ET_N_ESTIMATORS, max_depth=ET_MAX_DEPTH,
            min_samples_split=ET_MIN_SAMPLES_SPLIT,
            n_jobs=-1, random_state=RANDOM_SEED
        )
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.history = _pretrain_cnn(self.cnn, X_train, y_train, X_val, y_val,
                                     model_name='CNN-ExtraTrees')
        features = self.cnn.predict(X_train, batch_size=256)
        print(f"  CNN feature vector size: {features.shape[1]}")
        self.et.fit(features, y_train)
        return self

    def predict(self, X):
        return self.et.predict(self.cnn.predict(X, batch_size=256))

    def predict_proba(self, X):
        return self.et.predict_proba(self.cnn.predict(X, batch_size=256))

    def get_param_count(self):
        return self.cnn.count_params()

    def get_name(self):
        return "CNN-ExtraTrees"


# ============================================================
# MODEL 7: CNN-KNN
# ============================================================

class CNNKNN:
    """Lightweight CNN features + KNeighborsClassifier."""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.cnn = build_lightweight_cnn_feature_extractor(input_shape)
        self.scaler = StandardScaler()
        self.knn = None
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.neighbors import KNeighborsClassifier
        self.history = _pretrain_cnn(self.cnn, X_train, y_train, X_val, y_val,
                                     model_name='CNN-KNN')
        features = self.cnn.predict(X_train, batch_size=256)
        print(f"  CNN feature vector size: {features.shape[1]}")
        features_scaled = self.scaler.fit_transform(features)
        self.knn = KNeighborsClassifier(
            n_neighbors=KNN_N_NEIGHBORS, weights=KNN_WEIGHTS,
            metric=KNN_METRIC, n_jobs=-1
        )
        self.knn.fit(features_scaled, y_train)
        return self

    def predict(self, X):
        f = self.scaler.transform(self.cnn.predict(X, batch_size=256))
        return self.knn.predict(f)

    def predict_proba(self, X):
        f = self.scaler.transform(self.cnn.predict(X, batch_size=256))
        return self.knn.predict_proba(f)

    def get_param_count(self):
        return self.cnn.count_params()

    def get_name(self):
        return "CNN-KNN"


# ============================================================
# MODEL 8: CNN-XGBoost
# ============================================================

class XGBoostCNNRF:
    """Lightweight CNN features + XGBoost classifier."""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.cnn = build_lightweight_cnn_feature_extractor(input_shape)
        self.xgb = None
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from xgboost import XGBClassifier
        self.history = _pretrain_cnn(self.cnn, X_train, y_train, X_val, y_val,
                                     model_name='CNN-XGBoost')
        features = self.cnn.predict(X_train, batch_size=256)
        print(f"  CNN feature vector size: {features.shape[1]}")
        self.xgb = XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LR, use_label_encoder=False,
            eval_metric='mlogloss', n_jobs=-1, random_state=RANDOM_SEED,
            verbosity=1
        )
        print(f"  Fitting XGBoost on {len(features)} extracted features...")
        self.xgb.fit(features, y_train, verbose=True)
        print(f"  XGBoost fitting complete.")
        return self

    def predict(self, X):
        return self.xgb.predict(self.cnn.predict(X, batch_size=256))

    def predict_proba(self, X):
        return self.xgb.predict_proba(self.cnn.predict(X, batch_size=256))

    def get_param_count(self):
        return self.cnn.count_params()

    def get_name(self):
        return "CNN-XGBoost"


# ============================================================
# MODEL 9: CNN-LightGBM
# ============================================================

class LightGBMCNNRF:
    """Lightweight CNN features + LightGBM classifier."""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.cnn = build_lightweight_cnn_feature_extractor(input_shape)
        self.lgbm = None
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from lightgbm import LGBMClassifier
        self.history = _pretrain_cnn(self.cnn, X_train, y_train, X_val, y_val,
                                     model_name='CNN-LightGBM')
        features = self.cnn.predict(X_train, batch_size=256)
        print(f"  CNN feature vector size: {features.shape[1]}")
        self.lgbm = LGBMClassifier(
            n_estimators=LGBM_N_ESTIMATORS, max_depth=LGBM_MAX_DEPTH,
            learning_rate=LGBM_LR, random_state=RANDOM_SEED, verbose=1
        )
        print(f"  Fitting LightGBM on {len(features)} extracted features...")
        self.lgbm.fit(features, y_train)
        print(f"  LightGBM fitting complete.")
        return self

    def predict(self, X):
        return self.lgbm.predict(self.cnn.predict(X, batch_size=256))

    def predict_proba(self, X):
        return self.lgbm.predict_proba(self.cnn.predict(X, batch_size=256))

    def get_param_count(self):
        return self.cnn.count_params()

    def get_name(self):
        return "CNN-LightGBM"


# ============================================================
# MODEL 10: MobileNetV2 (DL baseline)
# ============================================================

class MobileNetV2Classifier:
    """MobileNetV2-style depthwise separable CNN (DL baseline)."""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self._build()

    def _build(self):
        inp = layers.Input(shape=self.input_shape)
        x = inp
        if AUGMENT_NOISE:
            x = layers.GaussianNoise(GAUSSIAN_NOISE_STDDEV)(x)

        # Project N bands → 3 channels for MobileNetV2 backbone
        x = layers.Conv2D(3, 1, activation='relu', name='band_projection',
                          kernel_regularizer=regularizers.l2(0.005))(x)
        x = layers.BatchNormalization()(x)
        if self.input_shape[0] < 32:
            x = layers.Resizing(32, 32)(x)

        for filters in [32, 64, 128]:
            x = layers.SeparableConv2D(filters, 3, padding='same', activation='relu',
                                       depthwise_regularizer=regularizers.l2(0.005))(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(N_CLASSES, activation='softmax')(x)

        self.model = models.Model(inp, x, name='MobileNetV2')
        self.model.compile(
            optimizer=AdamW(learning_rate=MOBILENET_LR),
            loss='categorical_crossentropy', metrics=['accuracy']
        )
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        cw = get_class_weights(y_train) if LW_CNN_CLASS_WEIGHTS else None
        y_smooth = tf.one_hot(y_train.astype(int), N_CLASSES)
        y_smooth = y_smooth * (1 - TRISAF_LABEL_SMOOTHING) + TRISAF_LABEL_SMOOTHING / N_CLASSES
        if X_val is not None:
            val_data = (X_val, tf.one_hot(y_val.astype(int), N_CLASSES))
        else:
            val_data = None
        self.history = self.model.fit(
            X_train, y_smooth, validation_data=val_data,
            epochs=MOBILENET_EPOCHS, batch_size=MOBILENET_BATCH_SIZE,
            callbacks=[
                callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5),
                TrainingLogger('MobileNetV2'),
            ], class_weight=cw, verbose=1
        )
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X, batch_size=256), axis=1)

    def predict_proba(self, X):
        return self.model.predict(X, batch_size=256)

    def get_param_count(self):
        return self.model.count_params()

    def get_name(self):
        return "MobileNetV2"


# ============================================================
# MODEL 11: RF-Only
# ============================================================

class RFOnly:
    """Random Forest on raw flattened patches (ML baseline)."""

    def __init__(self):
        self.rf = RandomForestClassifier(
            n_estimators=RF_ONLY_ESTIMATORS, max_depth=RF_ONLY_MAX_DEPTH,
            n_jobs=-1, random_state=RANDOM_SEED, verbose=1
        )
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"  Fitting RF on {len(X_train)} samples, {np.prod(X_train.shape[1:])} features...")
        self.rf.fit(X_train.reshape(len(X_train), -1), y_train)
        print(f"  RF fitting complete.")
        return self

    def predict(self, X):
        return self.rf.predict(X.reshape(len(X), -1))

    def predict_proba(self, X):
        return self.rf.predict_proba(X.reshape(len(X), -1))

    def get_param_count(self):
        return 0

    def get_name(self):
        return "RF Only"


# ============================================================
# MODEL 12: SVM-Only
# ============================================================

class SVMOnly:
    """SVM on raw flattened patches (ML baseline)."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.svm = None
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.svm import SVC
        X_flat = X_train.reshape(len(X_train), -1)
        if len(X_flat) > 5000:
            print(f"  WARNING: SVM on {len(X_flat)} samples with {X_flat.shape[1]} features"
                  " may be slow. Consider subsampling.")
        X_scaled = self.scaler.fit_transform(X_flat)
        self.svm = SVC(C=SVM_ONLY_C, kernel=SVM_ONLY_KERNEL, gamma=SVM_ONLY_GAMMA,
                       probability=True, random_state=RANDOM_SEED, verbose=True)
        print(f"  Fitting SVM on {len(X_scaled)} samples, {X_scaled.shape[1]} features...")
        self.svm.fit(X_scaled, y_train)
        print(f"  SVM fitting complete.")
        return self

    def predict(self, X):
        return self.svm.predict(self.scaler.transform(X.reshape(len(X), -1)))

    def predict_proba(self, X):
        return self.svm.predict_proba(self.scaler.transform(X.reshape(len(X), -1)))

    def get_param_count(self):
        return 0

    def get_name(self):
        return "SVM Only"


# ============================================================
# MODEL 13: XGBoost-Only
# ============================================================

class XGBoostOnly:
    """XGBoost on raw flattened patches (ML baseline)."""

    def __init__(self):
        self.xgb = None
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from xgboost import XGBClassifier
        X_flat = X_train.reshape(len(X_train), -1)
        self.xgb = XGBClassifier(
            n_estimators=XGB_ONLY_N_ESTIMATORS, max_depth=XGB_ONLY_MAX_DEPTH,
            learning_rate=XGB_ONLY_LR, use_label_encoder=False,
            eval_metric='mlogloss', n_jobs=-1, random_state=RANDOM_SEED,
            verbosity=1
        )
        print(f"  Fitting XGBoost on {len(X_flat)} samples, {X_flat.shape[1]} features...")
        self.xgb.fit(X_flat, y_train, verbose=True)
        print(f"  XGBoost fitting complete.")
        return self

    def predict(self, X):
        return self.xgb.predict(X.reshape(len(X), -1))

    def predict_proba(self, X):
        return self.xgb.predict_proba(X.reshape(len(X), -1))

    def get_param_count(self):
        return 0

    def get_name(self):
        return "XGBoost Only"


# ============================================================
# MODEL 14: ANN (MLP baseline)
# ============================================================

class ANNClassifier:
    """Multi-Layer Perceptron on flattened features (ML baseline)."""

    def __init__(self):
        self.mlp = MLPClassifier(
            hidden_layer_sizes=ANN_HIDDEN, activation='relu', solver='adam',
            max_iter=ANN_MAX_ITER, random_state=RANDOM_SEED,
            early_stopping=True, validation_fraction=0.15
        )
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.mlp.fit(X_train.reshape(len(X_train), -1), y_train)
        return self

    def predict(self, X):
        return self.mlp.predict(X.reshape(len(X), -1))

    def predict_proba(self, X):
        return self.mlp.predict_proba(X.reshape(len(X), -1))

    def get_param_count(self):
        return 0

    def get_name(self):
        return "ANN"


# ============================================================
# MODEL 15: EfficientNet-B0 (custom MBConv for small patches)
# ============================================================

def _mbconv_block(x, expand_filters, out_filters, stride=1, se_ratio=4):
    """Mobile inverted bottleneck with SE attention."""
    shortcut = x
    x = layers.Conv2D(expand_filters, 1, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(0.005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False,
                               depthwise_regularizer=regularizers.l2(0.005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    x = se_block(x, reduction=se_ratio)

    x = layers.Conv2D(out_filters, 1, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(0.005))(x)
    x = layers.BatchNormalization()(x)

    if stride == 1 and shortcut.shape[-1] == out_filters:
        x = layers.Add()([shortcut, x])
    return x


class EfficientNetB0Classifier:
    """Custom EfficientNet-B0 adapted for small multi-spectral patches."""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self._build()

    def _build(self):
        inp = layers.Input(shape=self.input_shape)
        x = inp
        if AUGMENT_NOISE:
            x = layers.GaussianNoise(GAUSSIAN_NOISE_STDDEV)(x)

        x = layers.Conv2D(EFFNET_FILTERS[0], 3, padding='same', use_bias=False,
                          kernel_regularizer=regularizers.l2(0.005))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)

        for i, f in enumerate(EFFNET_FILTERS[1:]):
            expand = x.shape[-1] * EFFNET_EXPAND_RATIO
            stride = 2 if (i % 2 == 0 and x.shape[1] is not None and x.shape[1] >= 4) else 1
            x = _mbconv_block(x, expand, f, stride=stride)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(N_CLASSES, activation='softmax')(x)

        self.model = models.Model(inp, x, name='EfficientNet_B0')
        self.model.compile(
            optimizer=AdamW(learning_rate=EFFNET_LR),
            loss='categorical_crossentropy', metrics=['accuracy']
        )
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        cw = get_class_weights(y_train)
        y_smooth = tf.one_hot(y_train.astype(int), N_CLASSES)
        y_smooth = y_smooth * (1 - TRISAF_LABEL_SMOOTHING) + TRISAF_LABEL_SMOOTHING / N_CLASSES
        if X_val is not None:
            val_data = (X_val, tf.one_hot(y_val.astype(int), N_CLASSES))
        else:
            val_data = None
        self.history = self.model.fit(
            X_train, y_smooth, validation_data=val_data,
            epochs=EFFNET_EPOCHS, batch_size=EFFNET_BATCH_SIZE,
            callbacks=[
                callbacks.EarlyStopping(patience=12, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5),
                TrainingLogger('EfficientNet-B0'),
            ], class_weight=cw, verbose=1
        )
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X, batch_size=256), axis=1)

    def predict_proba(self, X):
        return self.model.predict(X, batch_size=256)

    def get_param_count(self):
        return self.model.count_params()

    def get_name(self):
        return "EfficientNet-B0"


# ============================================================
# MODEL 16: Compact Vision Transformer (ViT-Tiny)
# ============================================================

class _PatchTokenize(layers.Layer):
    """Split spatial input into non-overlapping sub-patch tokens and project."""

    def __init__(self, token_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_size = token_size
        self.embed_dim = embed_dim
        self.proj = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(0.005))

    def call(self, x):
        bs = tf.shape(x)[0]
        h, w, c = x.shape[1], x.shape[2], x.shape[3]
        nh = h // self.token_size
        nw = w // self.token_size
        x = x[:, :nh * self.token_size, :nw * self.token_size, :]
        x = tf.reshape(x, [bs, nh, self.token_size, nw, self.token_size, c])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [bs, nh * nw, self.token_size * self.token_size * c])
        return self.proj(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'token_size': self.token_size, 'embed_dim': self.embed_dim})
        return cfg


class _TransformerBlock(layers.Layer):
    """Single transformer encoder block with pre-norm."""

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=embed_dim // num_heads,
                                             dropout=dropout)
        self.ln2 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu', kernel_regularizer=regularizers.l2(0.005)),
            layers.Dropout(dropout),
            layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(0.005)),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=None):
        norm = self.ln1(x)
        x = x + self.mha(norm, norm, training=training)
        norm = self.ln2(x)
        x = x + self.ffn(norm, training=training)
        return x


class _ClsPosEmbed(layers.Layer):
    """Prepend a learnable CLS token and add learnable positional embeddings.

    Uses add_weight so the parameters are tracked by the model (required for
    ModelCheckpoint/save_weights and stable graph execution on TF 2.17/Keras).
    Functionally identical to the previous tf.Variable-based implementation.
    """

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        num_tokens = int(input_shape[1])
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        self.cls_token = self.add_weight(
            name='cls_token', shape=(1, 1, self.embed_dim),
            initializer=init, trainable=True)
        self.pos_embed = self.add_weight(
            name='pos_embed', shape=(1, num_tokens + 1, self.embed_dim),
            initializer=init, trainable=True)
        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]
        cls = tf.broadcast_to(self.cls_token, [batch, 1, self.embed_dim])
        x = tf.concat([cls, x], axis=1)
        return x + self.pos_embed

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, self.embed_dim)


class CompactViTClassifier:
    """Compact Vision Transformer adapted for small multi-spectral patches."""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self._build()

    def _build(self):
        inp = layers.Input(shape=self.input_shape)
        x = inp
        if AUGMENT_NOISE:
            x = layers.GaussianNoise(GAUSSIAN_NOISE_STDDEV)(x)

        x = _PatchTokenize(VIT_PATCH_TOKEN, VIT_EMBED_DIM)(x)

        # CLS token + positional embedding as a tracked Keras layer (weights are
        # registered via add_weight so ModelCheckpoint/save_weights work correctly).
        x = _ClsPosEmbed(VIT_EMBED_DIM)(x)
        x = layers.Dropout(VIT_DROPOUT)(x)

        for _ in range(VIT_NUM_LAYERS):
            x = _TransformerBlock(VIT_EMBED_DIM, VIT_NUM_HEADS,
                                  VIT_MLP_DIM, VIT_DROPOUT)(x)

        x = layers.LayerNormalization()(x)
        cls_out = layers.Lambda(lambda t: t[:, 0])(x)
        out = layers.Dense(N_CLASSES, activation='softmax')(cls_out)

        self.model = models.Model(inp, out, name='Compact_ViT')
        self.model.compile(
            optimizer=AdamW(learning_rate=VIT_LR),
            loss='categorical_crossentropy', metrics=['accuracy']
        )
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        cw = get_class_weights(y_train)
        y_smooth = tf.one_hot(y_train.astype(int), N_CLASSES)
        y_smooth = y_smooth * (1 - TRISAF_LABEL_SMOOTHING) + TRISAF_LABEL_SMOOTHING / N_CLASSES
        if X_val is not None:
            val_data = (X_val, tf.one_hot(y_val.astype(int), N_CLASSES))
        else:
            val_data = None
        self.history = self.model.fit(
            X_train, y_smooth, validation_data=val_data,
            epochs=VIT_EPOCHS, batch_size=VIT_BATCH_SIZE,
            callbacks=[
                callbacks.EarlyStopping(patience=12, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5),
                TrainingLogger('Compact ViT'),
            ], class_weight=cw, verbose=1
        )
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X, batch_size=256), axis=1)

    def predict_proba(self, X):
        return self.model.predict(X, batch_size=256)

    def get_param_count(self):
        return self.model.count_params()

    def get_name(self):
        return "Compact ViT"
