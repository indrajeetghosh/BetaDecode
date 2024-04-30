import torch
from torch import nn
from torch.nn import functional as F


class CNNEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc_mean = nn.Linear(8192, latent_dim) # Adjusted the values
        self.fc_log_var = nn.Linear(8192, latent_dim)  # Adjusted the values

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, log_var

class TransposeCNNDecoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(TransposeCNNDecoder, self).__init__()
        self.fc_decode = nn.Linear(latent_dim, 256 * 1)  # Adjusted for initial reshape
        self.transpose_conv1 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.transpose_conv2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.transpose_conv3 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 256, 1)
        x =torch.relu(self.transpose_conv1(x))
        x = torch.relu(self.transpose_conv2(x))
        x = torch.sigmoid(self.transpose_conv3(x))
        return x

class BetaVAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(BetaVAE, self).__init__()
        self.encoder = CNNEncoder(input_shape, latent_dim)
        self.decoder = TransposeCNNDecoder(input_shape, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var