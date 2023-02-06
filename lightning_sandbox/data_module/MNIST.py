from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_lightning as pl

class MNIST (pl.LightningDataModule):
  def __init__(self, data_dir: str, batch_size: int, num_workers: int):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_workers = num_workers
    
  def prepare_data(self):
    datasets.MNIST(root=self.data_dir, train=True, download=True)
    datasets.MNIST(root=self.data_dir, train=False, download=True)
    
  def setup(self, stage=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if stage == "fit" or stage == None:
      train_dataset = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)
      self.train_dataset, self.val_dataset = random_split(train_dataset, [55000,5000])
    if stage == "test" or stage == None:
      self.test_dataset = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=transform)

  def train_dataloader(self):
    return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
  def val_dataloader(self):
    return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
  def test_dataloader(self):
    return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

