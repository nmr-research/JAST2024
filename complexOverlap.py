#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generating synthetic FIDs with peak overlap and variable noise
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import nmrglue as ng
import tensorflow as tf
from tensorflow import keras

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

def save_fig(filename, directory="images/robust_test", 
             tight_layout=True, fig_extension="png", resolution=300):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename + "." + fig_extension)
    print("Saving figure:", path)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def fid1D_generator(
    peak_count=100,
    noise_level=0.1,
    center_ppm=4.8,
    H_MHz=600,
    sw_ppm=14,
    TD=2048,
    R2rate=10 * np.pi * 2,
):
    shifts = np.random.normal(loc=8.5, scale=1.0, size=peak_count)
    offset_ppm = shifts - center_ppm
    freq_Hz = offset_ppm * H_MHz
    sw_Hz = sw_ppm * H_MHz
    dt = 1.0 / sw_Hz  # spacing 
    time = np.linspace(0, dt * (TD - 1), TD)
    R2_array = np.random.normal(R2rate, 1.0, peak_count)
    fid_clean = np.zeros(TD, dtype=complex)
    for i in range(peak_count):
        fid_component = np.exp(1j * 2 * np.pi * freq_Hz[i] * time) * np.exp(
            -(1.0 + 0.j) * R2_array[i] * time
        )
        fid_clean += fid_component
    noise_real = (np.random.rand(TD) - 0.5) * noise_level
    noise_imag = (np.random.rand(TD) - 0.5) * noise_level
    fid_noisy = fid_clean + (noise_real + 1j * noise_imag) * np.mean(np.abs(fid_clean))
    return fid_noisy, fid_clean
  
def ft_and_process(fid):
    pdata = fid.copy()
    pdata[0] *= 0.5
    pdata = ng.proc_base.sp(pdata, off=0.5, end=0.95, pow=2.0)
    pdata = ng.proc_base.zf_size(pdata, len(fid))  
    pdata = ng.proc_base.fft(pdata)
    pdata = ng.proc_base.di(pdata)
    max_val = np.max(np.abs(pdata))
    pdata /= (max_val if max_val != 0 else 1.0)
    return pdata

def augment_fid(fid, phase_range=0.1, scale_range=0.1):
    """Apply random phase rotation and scaling"""
    phase = np.random.uniform(-phase_range, phase_range)
    scale = 1.0 + np.random.uniform(-scale_range, scale_range)
    return fid * scale * np.exp(1j * phase)

num_fid = 10000  
TD = 2048
sw_ppm = 14
H_MHz = 600
center_ppm = 4.8

X_fids = np.zeros((num_fid, TD), dtype=complex)
Y_specs = np.zeros((num_fid, TD))  

for i in range(num_fid):
    n_peaks = np.random.randint(low=50, high=201)
    n_level = np.random.uniform(0.1, 0.5)
    noisy_fid, clean_fid = fid1D_generator(
        peak_count=n_peaks,
        noise_level=n_level,
        center_ppm=center_ppm,
        H_MHz=H_MHz,
        sw_ppm=sw_ppm,
        TD=TD,
    )
    spec = ft_and_process(noisy_fid)
    max_val = np.max(np.abs(noisy_fid))
    norm_fid = noisy_fid / max_val if max_val != 0 else noisy_fid
    augmented_fid = augment_fid(norm_fid)
    X_fids[i] = augmented_fid
    Y_specs[i] = ft_and_process(augmented_fid)

split_train = int(num_fid * 0.8)
split_valid = int(num_fid * 0.9)

X_train = X_fids[:split_train]
X_valid = X_fids[split_train:split_valid]
X_test  = X_fids[split_valid:]

Y_train = Y_specs[:split_train]
Y_valid = Y_specs[split_train:split_valid]
Y_test  = Y_specs[split_valid:]

def reshape_fids_to_real_imag(fid_array):
    """
    Takes shape (N, TD) complex => shape (N, 2*TD) real & imag.
    """
    N, TD_ = fid_array.shape
    out = np.zeros((N, 2 * TD_), dtype=np.float32)
    out[:, 0::2] = fid_array.real
    out[:, 1::2] = fid_array.imag
    return out

X_train_ri = reshape_fids_to_real_imag(X_train)
X_valid_ri = reshape_fids_to_real_imag(X_valid)
X_test_ri  = reshape_fids_to_real_imag(X_test)

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(2*TD,)),
    keras.layers.Dense(4096, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2048, activation="elu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2048, activation="elu"),
    keras.layers.Dense(TD)
])

initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.9

def spectral_loss(y_true, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    peak_loss = tf.reduce_mean(tf.abs(tf.reduce_max(y_true, axis=1) - 
                                    tf.reduce_max(y_pred, axis=1)))
    return mse + 0.1 * peak_loss

model.compile(optimizer='adam', loss=spectral_loss)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss=spectral_loss)
early_stop = keras.callbacks.EarlyStopping(
    patience=10, 
    restore_best_weights=True,
    min_delta=0.0001 
)

history = model.fit(
    X_train_ri, Y_train,
    validation_data=(X_valid_ri, Y_valid),
    epochs=200,  
    batch_size=64,  
    callbacks=[early_stop],
    verbose=1
)

Y_pred_test = model.predict(X_test_ri)
errors = Y_pred_test - Y_test
