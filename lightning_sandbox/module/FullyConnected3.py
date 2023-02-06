import torch.nn as nn
import torch.optim as Optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

class FullyConnected3(pl.LightningModule):
  def __init__(self, learning_rate: float):
    super().__init__()
    # hyperparameters
    self.save_hyperparameters()
    # nets
    self.fc1 = nn.Linear(28*28,28*28)
    self.fc2 = nn.Linear(28*28,28*28)
    self.fc3 = nn.Linear(28*28,10)
  def forward(self,x):
    x = x.flatten(start_dim=1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    out = self.fc3(x)
    return out
  
  def training_step(self, batch, batch_idx):
    x,y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)
    acc = accuracy(y_hat, y, task='multiclass', num_classes=10)
    metrics = {"train_loss":loss, "train_acc":acc}
    self.log_dict(metrics)
    return loss
  
  def validation_step(self, batch, batch_idx):
    x,y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)
    acc = accuracy(y_hat, y, task='multiclass', num_classes=10)
    metrics = {"val_loss":loss, "val_acc":acc}
    self.log_dict(metrics)
  
  def test_step(self, batch, batch_idx):
    x,y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)
    acc = accuracy(y_hat, y, task='multiclass', num_classes=10)
    metrics = {"test_loss":loss, "test_acc":acc}
    self.log_dict(metrics)
  
  def configure_optimizers(self):
    return Optim.Adam(self.parameters(),lr=self.hparams.learning_rate)
