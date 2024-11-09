import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import itertools
#import seaborn as sns

import matplotlib as mpl
import pylab
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['lines.color'] = 'r'
mpl.rcParams['font.weight'] = 200
plt.style.use('seaborn-whitegrid')
plt.rc('figure',figsize=(20,14))
mpl.axes.Axes.annotate
mpl.rcParams['font.family'] = "serif"
pylab.rcParams['ytick.major.pad']='15'
pylab.rcParams['xtick.major.pad']='15'
mpl.rcParams['font.weight'] = "semibold"
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['axes.linewidth'] = 4
mpl.rcParams['xtick.labelsize'] = 30
mpl.rcParams['ytick.labelsize'] = 30
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['legend.fontsize'] = 30

def plot_tsne_per_channel(encoded_val, encoded_denoised, num_channels, save_path="Images"):
    for channel in range(num_channels):
        val_channel_data = encoded_val[:, channel, :].reshape(encoded_val.shape[0], -1)
        denoised_channel_data = encoded_denoised[:, channel, :].reshape(encoded_denoised.shape[0], -1)

        combined_data = np.vstack([val_channel_data, denoised_channel_data])

        tsne = TSNE(n_components=2, random_state=42, verbose=1, perplexity=50, n_iter=1000)
        combined_tsne = tsne.fit_transform(combined_data)

        val_channel_tsne = combined_tsne[:val_channel_data.shape[0]]
        denoised_channel_tsne = combined_tsne[val_channel_data.shape[0]:]

        plt.figure(figsize=(8, 8))
        plt.scatter(val_channel_tsne[:, 0], val_channel_tsne[:, 1], label='Noisy Features', alpha=0.4, c='orange', marker='o')
        plt.scatter(denoised_channel_tsne[:, 0], denoised_channel_tsne[:, 1], label='Denoised Features', alpha=0.4, c='green', marker='s')
        
        plt.xlabel('Embedding Dimension 1', fontweight='semibold')
        plt.ylabel('Embedding Dimension 2', fontweight='semibold')
        plt.legend(fontsize=12, loc='best')
        plt.title(f't-SNE Visualization of Channel {channel + 1}', fontsize=16)

        plt.savefig(f"{save_path}/Denoise_features_channel_{channel + 1}.png", format='png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()