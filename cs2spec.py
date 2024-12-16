# -*- coding: utf-8 -*-
"""
This script simulates 1D NMR FIDs from given chemical shifts, processes them 
into spectra, and uses neural networks (via TensorFlow/Keras) to model the 
FID-to-spectrum mapping. It also demonstrates how to derive peak lists from 
predicted spectra.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import nmrglue as ng
import tensorflow as tf
import tensorflow.keras as keras

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "noFT"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """Save a figure with given parameters."""
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def fid1D_generator_from_chemical_shift(chemical_shifts, center_ppm=4.8, H_MHz=600, sw_ppm=14, TD=2048, R2rate=10*np.pi*2):
    """
    Generate a 1D FID from given chemical shifts.
    
    Parameters:
    chemical_shifts : 1D array of chemical shifts (in ppm)
    center_ppm : Reference offset (in ppm)
    H_MHz : Spectrometer frequency in MHz
    sw_ppm : Spectral width in ppm
    TD : Number of data points
    R2rate : Transverse relaxation rate (in s^-1)
    
    Returns:
    fidnoise, fid : Complex arrays representing noisy FID and noiseless FID
    """
    num_residue = len(chemical_shifts)
    Hppmnormoffseted = chemical_shifts - center_ppm
    Hfreq = Hppmnormoffseted * H_MHz

    sw_Hz = sw_ppm * H_MHz
    dt = 1/sw_Hz
    acq = dt * TD / 2
    time = np.linspace(0, acq, int(TD/2))

    R2 = np.random.normal(R2rate, 1, num_residue)
    fid = np.empty((num_residue, int(TD/2)), dtype=complex)
    fidnoise = np.empty((num_residue, int(TD/2)), dtype=complex)

    for i in range(len(Hfreq)):
        fidind = np.exp(1.j * 2 * np.pi * Hfreq[i] * time) * np.exp(-(1+0.j)*R2[i]*time)
        fidindnoise = fidind + (1 + 1j)*0.1*(np.random.rand(1, int(TD/2)) - 0.5)
        fid[i, :] = fidind
        fidnoise[i, :] = fidindnoise

    fid = np.sum(fid, axis=0)
    fidnoise = np.sum(fidnoise, axis=0)
    return fidnoise, fid

def processing(data):
    """
    Process the input time-domain data using standard NMR procedures:
    - First point scaling
    - Apodization (sine-squared window)
    - Zero-filling
    - FFT
    - Discard imaginary part
    """
    pdata = data.copy()
    pdata[0] = pdata[0]*0.5
    pdata = ng.proc_base.sp(pdata, off=0.5, end=0.95, pow=2.0)
    pdata = ng.proc_base.zf(pdata, len(pdata))
    pdata = ng.proc_base.fft(pdata)
    pdata = ng.proc_base.di(pdata)
    return pdata

if __name__ == "__main__":
    num_fid = 10000
    TD = 2048
    num_residue = 100
    center_ppm = 4.8
    sw_ppm = 14
    H_MHz = 600
    
    x_fids = np.empty((num_fid, int(TD/2)), dtype=complex)
    y_fids = np.empty((num_fid, int(TD/2)), dtype=complex)
    x_specs = np.empty((num_fid, TD))
    y_specs = np.empty((num_fid, TD))
    Hppms = np.empty((num_fid, num_residue))

    for a in range(num_fid):
        np.random.seed(a)
        Hppm = np.random.normal(8.5, 1, num_residue)
        Hppm = np.round(Hppm, 2)
        x_fid, y_fid = fid1D_generator_from_chemical_shift(Hppm, TD=TD)
        x_spec = processing(x_fid)
        y_spec = processing(y_fid)

        x_fid = x_fid / np.max(np.abs(x_fid))
        y_fid = y_fid / np.max(np.abs(y_fid))
        x_spec = x_spec / np.max(np.abs(x_spec))
        y_spec = y_spec / np.max(np.abs(y_spec))

        x_fids[a] = x_fid
        y_fids[a] = y_fid
        x_specs[a] = x_spec
        y_specs[a] = y_spec
        Hppms[a] = Hppm

    Hppms.sort()

    Hppmmin = center_ppm - sw_ppm/2
    Hppmmax = center_ppm + sw_ppm/2

    Hindex = (TD * (Hppms - Hppmmin) // sw_ppm).astype(int)
    Hppmfull = np.zeros((Hindex.shape[0], TD))
    for i in range(Hindex.shape[0]):
        for idx in Hindex[i, :]:
            if idx < TD:
                Hppmfull[i, idx] += 1

    x_train = Hppmfull[:8000]
    x_valid = Hppmfull[8000:9000]
    x_test = Hppmfull[9000:]

    y_train = y_specs[:8000]
    y_valid = y_specs[8000:9000]
    y_test = y_specs[9000:]

    # Example training
    keras.backend.clear_session()
    np.random.seed(44)
    tf.random.set_seed(44)

    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(2048,)),
        keras.layers.Dense(2048)
    ])

    model.compile(loss='mse', optimizer='adam')
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
    history = model.fit(x_train, y_train,
                        validation_data=(x_valid, y_valid),
                        callbacks=callbacks,
                        epochs=1000)

    # Example evaluation
    y_pred_test = model.predict(x_test)
    plt.plot(y_pred_test[0, 1024:], 'k', label='Predicted')
    plt.plot(x_specs[9000, 1024:], 'r', label='Reference')
    plt.xlabel('Index')
    plt.ylabel('Intensity')
    plt.legend()
    save_fig("comparison_predicted_vs_ref")
    plt.show()

    test_loss = model.evaluate(x_test, y_test)
    print("Test Loss:", test_loss)
