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

class VAE_bce (pl.LightningModule):
  def __init__(self, learning_rate:float, input_dim:int, latent_dim:int):
    super().__init__()

    self.save_hyperparameters()
    
    # buffers used to create prior_z
    self.register_buffer("zero_mean", torch.zeros(self.hparams.latent_dim))
    self.register_buffer("one_std", torch.ones(self.hparams.latent_dim))
    
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
    self.prob_decoder = nn.Sequential(
      DecoderModulo(256,256,8),
      nn.ConvTranspose2d(256, self.hparams.input_dim, kernel_size=4, stride=2, padding=1),
      nn.Sigmoid() # image should have range [0..1]
    )
    self._init_parameters()

  def _init_parameters(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

  def _get_mlv_encoder(self, x):
    x = self.encoder(x)
    mean, lgvar = self.mean_encoder(x), self.lgvar_encoder(x)
    return mean, lgvar

  def _get_mlv_decoder(self, x):
    x = self.decoder(x)
    prob = self.prob_decoder(x)
    return prob

  def _shared_step(self, batch, batch_idx):
    x, _ = batch
    batch_size = x.shape[0]
  
    mean_z, lgvar_z = self._get_mlv_encoder(x)
    dist_z = Normal(mean_z, (lgvar_z/2).exp())
    z = dist_z.rsample()

    prob_xhat = self._get_mlv_decoder(z)
    
    # KL-Divergence
    prior_z = Normal(self.zero_mean, self.one_std)
    kl_term = kl_divergence(dist_z, prior_z).sum()
    
    # Negative log likelihood
    reconst_term = F.binary_cross_entropy(prob_xhat, x, reduction="sum")

    # Negative ELBO
    n_ELBO = (kl_term + reconst_term) / batch_size # per batch
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

  def predict_step(self, batch, batch_idx, z=None):
    # use image to generate similar image
    x, _ = batch
    assert x.shape[0] == 1, "Predict mode only accept batch size 1"

    if z == None:
      mean_z, lgvar_z = self._get_mlv_encoder(x)
      dist_z = Normal(mean_z, (lgvar_z/2).exp())
      z = dist_z.sample().squeeze()
    z = z.unsqueeze(0)
    
    xhat = self._get_mlv_decoder(z)
    return xhat.squeeze().detach()

  def generate_Img(self, z=None):
    # generate new image from Gaussian Noise
    if z == None:
      prior_z = Normal(self.zero_mean, self.one_std)
      z = prior_z.sample()
    z = z.unsqueeze(0)

    xhat = self._get_mlv_decoder(z)
    return xhat.squeeze().detach()

  def configure_optimizers(self):
    optimizer = Optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    lr_scheduler = Optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}









