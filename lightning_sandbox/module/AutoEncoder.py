import torch.nn as nn
import torch.optim as Optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy



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

class AutoEncoder (pl.LightningModule):
  def __init__(self, learning_rate:float, output_dim:int):
    super().__init__()
    self.save_hyperparameters()
    self.encoder = nn.Sequential(
      EncoderModulo(3,256,8),
      EncoderModulo(256,256,8),
      EncoderModulo(256,256,8),
      EncoderModulo(256,256,8),
      nn.Conv2d(256, self.hparams.output_dim, kernel_size=4, padding=1, stride=2),
      nn.Flatten()
    )
    self.decoder = nn.Sequential(
      nn.Unflatten(1, (self.hparams.output_dim,1,1)),
      DecoderModulo(self.hparams.output_dim,256,8),
      DecoderModulo(256,256,8),
      DecoderModulo(256,256,8),
      DecoderModulo(256,256,8),
      nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1)
    )

  def forward(self, x):
    out = self.encoder(x)
    return out

  def on_train_start(self):
    self.logger.log_hyperparams(self.hparams, {"hp/train_loss": 0, "hp/test_loss": 0})

  def training_step(self, batch, batch_idx):
    x, _ = batch
    xhat = self(x)
    xhat = self.decoder(xhat)
    loss = F.mse_loss(x, xhat)
    metrics = {'train_loss':loss, "hp/train_loss": loss}
    self.log_dict(metrics)
    return loss

  def validation_step(self, batch, batch_idx):
    x, _ = batch
    xhat = self(x)
    xhat = self.decoder(xhat)
    loss = F.mse_loss(x, xhat)
    metrics = {'val_loss':loss, "hp/val_loss": loss}
    self.log_dict(metrics)
  
  def test_step(self, batch, batch_idx):
    x, _ = batch
    xhat = self(x)
    xhat = self.decoder(xhat)
    loss = F.mse_loss(x, xhat)
    self.log('test_loss', loss)
    return x, xhat

  def configure_optimizers(self):
    return Optim.Adam(self.parameters(), lr=self.hparams.learning_rate)









