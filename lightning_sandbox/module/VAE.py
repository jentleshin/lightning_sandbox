import torch
import torch.nn as nn
import torch.optim as Optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.distributions import Normal, LogNormal, kl_divergence



class EncoderModulo (nn.Module):
  def __init__(self, input_dim:int, output_dim:int, group_num:int):
    super().__init__()
    self.model = nn.Sequential(
      nn.Conv2d(input_dim, output_dim, kernel_size=4, padding=1, stride=2),
      nn.GroupNorm(group_num, output_dim),
      nn.GELU()
    )
  def forward(self, x):
    out = self.model(x)
    return out

class DecoderModulo (nn.Module):
  def __init__(self, input_dim:int, output_dim:int, group_num:int):
    super().__init__()
    self.model = nn.Sequential(
      nn.ConvTranspose2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1),
      nn.GroupNorm(group_num, output_dim),
      nn.GELU()
    )
  def forward(self, x):
    out = self.model(x)
    return out

class VAE (pl.LightningModule):
  def __init__(self, learning_rate:float, input_dim:int, latent_dim:int):
    super().__init__()

    self.save_hyperparameters()

    self.encoder = nn.Sequential(
      EncoderModulo(self.hparams.input_dim,256,8),
      EncoderModulo(256,256,8),
      EncoderModulo(256,256,8),
    )
    self.mean_encoder = nn.Sequential(
      EncoderModulo(256,256,8),
      nn.Conv2d(256, self.hparams.latent_dim, kernel_size=4, padding=1, stride=2),
      nn.Flatten()
    )
    self.lgvar_encoder = nn.Sequential(
      EncoderModulo(256,256,8),
      nn.Conv2d(256, self.hparams.latent_dim, kernel_size=4, padding=1, stride=2),
      nn.Flatten()
    )
    self.decoder = nn.Sequential(
      nn.Unflatten(1, (self.hparams.latent_dim,1,1)),
      DecoderModulo(self.hparams.latent_dim,256,8),
      DecoderModulo(256,256,8),
      DecoderModulo(256,256,8),
    )
    self.mean_decoder = nn.Sequential(
      DecoderModulo(256,256,8),
      nn.ConvTranspose2d(256, self.hparams.input_dim, kernel_size=4, stride=2, padding=1)
    )
    self.lgvar_decoder = nn.Sequential(
      DecoderModulo(256,256,8),
      nn.ConvTranspose2d(256, self.hparams.input_dim, kernel_size=4, stride=2, padding=1),
    )
    
  def _get_mlv_encoder(self, x):
    x = self.encoder(x)
    mean, lgvar = self.mean_encoder(x), self.lgvar_encoder(x)
    return mean, lgvar

  def _get_mlv_decoder(self, x):
    x = self.decoder(x)
    mean, lgvar = self.mean_decoder(x), self.lgvar_decoder(x)
    return mean, lgvar

  def _shared_step(self, batch, batch_idx):
    x, _ = batch
    batch_size = x.shape[0]
    
    mean_z, lgvar_z = self._get_mlv_encoder(x)
    dist_z = Normal(mean_z, (lgvar_z/2).exp())
    z = dist_z.rsample()

    mean_xhat, lgvar_xhat = self._get_mlv_decoder(z)
    dist_xhat = Normal(mean_xhat, (lgvar_xhat/2).exp())

    prior_z = Normal(torch.zeros(self.hparams.latent_dim).to(x), torch.ones(self.hparams.latent_dim).to(x))
    kl_term = kl_divergence(dist_z, prior_z).sum()
    nll_term = F.gaussian_nll_loss(mean_xhat, x, lgvar_xhat.exp(), eps=1e-6, reduction="sum") # mathematically stable version of nll_loss
    n_ELBO = (kl_term + nll_term) / batch_size # loss per batch
    return n_ELBO

  def on_train_start(self):
    self.logger.log_hyperparams(self.hparams, {"hp/train_negative_ELBO": 0, "hp/val_negative_ELBO": 0})

  def training_step(self, batch, batch_idx):
    n_ELBO = self._shared_step(batch, batch_idx)
    metrics = {'train_negative_ELBO':n_ELBO, "hp/train_negative_ELBO": n_ELBO}
    self.log_dict(metrics)
    return n_ELBO

  def validation_step(self, batch, batch_idx):
    n_ELBO = self._shared_step(batch, batch_idx)
    metrics = {'val_negative_ELBO':n_ELBO, "hp/val_negative_ELBO": n_ELBO}
    self.log_dict(metrics)
  
  def test_step(self, batch, batch_idx):
    n_ELBO = self._shared_step(batch, batch_idx)
    metrics = {'test_negative_ELBO':n_ELBO, "hp/test_negative_ELBO": n_ELBO}
    self.log_dict(metrics)

  def predict_step(self, batch, batch_idx):
    x, _ = batch
    mean_z, lgvar_z = self._get_mlv_encoder(x)
    dist_z = Normal(mean_z, (lgvar_z/2).exp())
    z = dist_z.sample()

    mean_xhat, lgvar_xhat = self._get_mlv_decoder(z)
    dist_xhat = Normal(mean_xhat, (lgvar_xhat/2).exp())
    xhat = dist_xhat.sample()
    return x, xhat

  def configure_optimizers(self):
    return Optim.Adam(self.parameters(), lr=self.hparams.learning_rate)









