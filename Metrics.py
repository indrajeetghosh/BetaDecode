#Denoising EEG Signals for Real-World BCI Applications Using GANs
#EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG denoising

import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import signal
from scipy.stats import pearsonr

def rms(signal):
    return torch.sqrt(torch.mean(signal ** 2, dim=-1))

def compute_rrmse_temporal(f_y, x):
    rrmse_numerator = rms(f_y - x)
    rrmse_denominator = rms(x)
    rrmse_temporal = rrmse_numerator / (rrmse_denominator + 1e-8)
    return rrmse_temporal.mean(dim=0)  

def compute_rrmse_spectral(f_y, x, fs):
    # Compute (PSD for each channel using Welch's method
    psd_f_y = []
    psd_x = []
    for b in range(f_y.size(0)):
        psd_f_y_batch = []
        psd_x_batch = []
        for ch in range(f_y.size(1)):
            f, psd_f_y_ch = signal.welch(f_y[b, ch, :].cpu().numpy(), fs=fs, nperseg=fs)
            f, psd_x_ch = signal.welch(x[b, ch, :].cpu().numpy(), fs=fs, nperseg=fs)
            psd_f_y_batch.append(psd_f_y_ch)
            psd_x_batch.append(psd_x_ch)
        psd_f_y.append(psd_f_y_batch)
        psd_x.append(psd_x_batch)

    psd_f_y = torch.tensor(psd_f_y)
    psd_x = torch.tensor(psd_x)

    rrmse_numerator = rms(psd_f_y - psd_x)
    rrmse_denominator = rms(psd_x)
    rrmse_spectral = rrmse_numerator / (rrmse_denominator + 1e-8)
    return rrmse_spectral.mean(dim=0)  

def compute_pearson_correlation(f_y, x):
    f_y = f_y.cpu().numpy() 
    x = x.cpu().numpy()

    cc_per_channel = []
    for ch in range(f_y.shape[1]):
        channel_corrs = [pearsonr(f_y[b, ch, :], x[b, ch, :])[0] for b in range(f_y.shape[0])]
        cc_per_channel.append(np.mean(channel_corrs)) 

    cc_avg = np.mean(cc_per_channel)  
    return cc_per_channel, cc_avg