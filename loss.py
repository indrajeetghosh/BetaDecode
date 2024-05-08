from torch.nn.functional import cosine_similarity
from torch.nn import functional as F

def kl_loss(mu, logvar):
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    return KLD

def reconstruction_loss(x_reconstructed, x):
    cos_sim_loss = 1 - cosine_similarity(x, x_reconstructed, dim=1).mean()
    mse = F.mse_loss(x_reconstructed, x).mean()
    recon = cos_sim_loss + mse
    return recon

def vae_loss(y_pred, y_true):
    mu, logvar, recon_x = y_pred
    recon_loss = reconstruction_loss(recon_x, y_true)
    kld_loss = vae_gaussian_kl_loss(mu, logvar)
    total_loss = 5 * recon_loss + 5 * kld_loss
    return total_loss
