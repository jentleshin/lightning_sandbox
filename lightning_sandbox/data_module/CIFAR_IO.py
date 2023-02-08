from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_lightning as pl

class CIFAR_IO (pl.LightningDataModule):
  def __init__(self, data_dir: str, batch_size: int, num_workers: int):
    super().__init__()
    self.save_hyperparameters()
  
  def prepare_data(self):
    datasets.CIFAR10(root=self.hparams.data_dir, train=True, download=True)
    datasets.CIFAR10(root=self.hparams.data_dir, train=False, download=True)
  
  def setup(self, stage: str):
    transform = transforms.Compose([
      transforms.ToTensor(),
      #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    if stage == "fit" or stage == None:
      dataset = datasets.CIFAR10(root=self.hparams.data_dir, train=True, transform=transform, download=False)
      self.train_dataset, self.val_dataset = random_split(dataset, [49000,1000])
    elif stage == "test" or stage == None:
      self.test_dataset = datasets.CIFAR10(root=self.hparams.data_dir, train=False, transform=transform, download=False)
    elif stage == "predict" or stage == None:
      self.predict_dataset = datasets.CIFAR10(root=self.hparams.data_dir, train=False, transform=transform, download=False)
  
  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)
  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)
  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)
  def predict_dataloader(self):
    return DataLoader(self.predict_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)
