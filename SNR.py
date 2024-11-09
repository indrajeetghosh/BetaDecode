import torch

def calculate_rms(signal):
    signal_tensor = torch.tensor(signal)
    return torch.sqrt(torch.mean(signal ** 2))

def calculate_snr(clean_signal, augmented_signal):
    rms_clean = calculate_rms(clean_signal)
    rms_augmented = calculate_rms(augmented_signal)
    print("RMS of clean signal:", rms_clean.item())
    print("RMS of augmented signal:", rms_augmented.item())
    snr = 10 * torch.log10(rms_clean / rms_augmented)
    return snr.item()