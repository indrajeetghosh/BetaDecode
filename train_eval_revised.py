#first install- !pip install torch braindecode

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.nn.functional import cosine_similarity
from torch.optim.lr_scheduler import ReduceLROnPlateau
from numpy import linspace
from braindecode.augmentation import FTSurrogate, SmoothTimeMask, ChannelsDropout

import loss 
import DataLoader
import ModelNoAtt
import ModelAtt
from DataLoader import DenoiseDataset
from SNR import calculate_snr, calculate_rms
from Aug import apply_augmentation
from Normalization import normalize


sfreq = 1200
alpha = 0.05
seed = 42

transforms_freq = [FTSurrogate(probability=0.05, phase_noise_magnitude=phase_freq,
                               random_state=seed) for phase_freq in linspace(0, 0.2, 1)]

transforms_time = [SmoothTimeMask(probability=0.05, mask_len_samples=int(sfreq * second),
                                  random_state=seed) for second in linspace(0.1, 0.2, 1)]

transforms_spatial = [ChannelsDropout(probability=0.05, p_drop=prob,
                                      random_state=seed) for prob in linspace(0, 0.2, 1)]

tranform_gaussian = [GaussianNoise(probability = 0.05, std=0.1, random_state=None)]

# Load EEG data and labels
EEG_feature = np.load('DATA/ori_EEG32.npy')
EEG_Labels = np.load('DATA/ori_Labels.npy')

EEG_feature = EEG_feature.reshape(EEG_feature.shape[2], EEG_feature.shape[1], EEG_feature.shape[0])
EEG_feature = normalize(EEG_feature)

# Check the shapes of EEG data and labels
print("EEG_feature shape:", EEG_feature.shape)
print("EEG_Labels shape:", EEG_Labels.shape)

unique_labels, label_counts = np.unique(EEG_Labels, return_counts=True)
print("Unique labels:", unique_labels)
print("Label counts:", label_counts)

indices = np.where(EEG_Labels != 14)[0]

# Filter out those samples from both features and labels
EEG_feature_filtered = EEG_feature[indices]
labels_filtered = EEG_Labels[indices]

noise_indices = np.where(EEG_Labels == 14)[0]
noise = EEG_feature[noise_indices]

print(EEG_feature_filtered.shape, labels_filtered.shape)
# Check the shape of filtered data and labels
print("Filtered EEG_feature shape:", EEG_feature_filtered.shape)
print("Filtered labels shape:", labels_filtered.shape)
unique_labels_filtered, unique_labels_filtered_counts = np.unique(labels_filtered, return_counts=True)
print("Unique labels filtered:", unique_labels_filtered)
print("Unique labels filtered counts:", unique_labels_filtered_counts)


X_train, X_test, y_train, y_test = train_test_split(EEG_feature_filtered, labels_filtered, test_size=0.1, random_state=42)

# Check the shapes of training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

noisy_samples_train = apply_augmentation(X_train, alpha, transforms=transforms_freq + transforms_time + transforms_spatial + tranform_gaussian)
noisy_samples_test = apply_augmentation(X_test, alpha, transforms=transforms_freq + transforms_time + transforms_spatial + tranform_gaussian)

batch_size = 10

train_dataset = DenoiseDataset(x_train)#(clean_noise)
test_dataset = DenoiseDataset(x_test)#(test_clean_noise)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, shuffle=True)

# Check the length of the loaders
print("Number of batches in train loader:", len(train_loader))
print("Number of batches in eval loader:", len(test_loader))

dataiter = iter(train_dataset)
dataiter1 = iter(test_dataset)
data = next(dataiter)
#data1 = next(dataiter1)

print(type(data))

input_shape = (32, 1200)
latent_dim = 32
epochs = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BetaVAE(input_shape=input_shape, latent_dim=latent_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00003, betas=(0.5, 0.9), weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor = 0.15)
total_loss_list = []
for epoch in range(epochs):
    beta_vae_model.train()
    total_loss = 0
    for batch_idx, data in enumerate(train_loader):
        inputs = data.float().to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = beta_vae_model(inputs)
        recon_batch = recon_batch.permute(0, 2, 1)
        loss = vae_loss((mu, log_var, recon_batch), inputs)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        #scheduler.step(total_loss)    
    epoch_loss = total_loss / len(train_loader)
    total_loss_list.append(epoch_loss)
    print('Epoch {}, Loss: {:.4f}'.format(epoch+1, epoch_loss))
    
    
    
model.eval()

original_signals = []
reconstructed_signals = []

with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        inputs = data.float().to(device)
        recon_batch, mean, variance = beta_vae_model(inputs)
        original_signals.append(inputs.cpu().numpy())
        reconstructed_signals.append(recon_batch.cpu().numpy())
original_signals = np.concatenate(original_signals, axis=0)
reconstructed_signals = np.concatenate(reconstructed_signals, axis=0)
reconstructed_signals = np.transpose(reconstructed_signals, (0, 2, 1))

rms_n = np.sqrt(np.mean(np.square(reconstructed_signals - original_signals)))
rms_d = np.sqrt(np.mean(np.square(original_signals)))
rrmse_temporal = rms_n / rms_d

original_signals_flattened = original_signals.reshape(-1)
reconstructed_signals_flattened = reconstructed_signals.reshape(-1)

#print("RRMSE Temporal:", rrmse_temporal)


# Pearson Coefficient

channel_correlations = []

for channel in range(original_signals.shape[2]):
    # Extract the channel data from both original and reconstructed signals
    original_channel_data = original_signals[:, :, channel].flatten()  # Flatten to make 1D
    reconstructed_channel_data = reconstructed_signals[:, :, channel].flatten()  # Flatten to make 1D

    # Compute the Pearson correlation coefficient for this channel
    correlation, _ = pearsonr(original_channel_data, reconstructed_channel_data)
    
    # Append the correlation coefficient to the list
    channel_correlations.append(correlation)

# Compute the average of the Pearson correlation coefficients across all channels
average_correlation = np.mean(channel_correlations)

#print(f"Average Pearson Correlation Coefficient across channels: {average_correlation}")

data = {
    'RRMSE Temporal': [rrmse_temporal],
    'Pearson Correlation Coefficient': [average_correlation]
}
df = pd.DataFrame(data)
print("Metrics saved to metrics.csv")
