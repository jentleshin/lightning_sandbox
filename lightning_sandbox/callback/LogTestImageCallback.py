import pytorch_lightning as pl
from torchvision.utils import make_grid

class LogTestImageCallback (pl.Callback):
  def __init__(self):
    super().__init__()
  
  def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    x, xhat = outputs
    trainer.logger.experiment.add_image(f"{batch_idx}/input",make_grid(x))
    trainer.logger.experiment.add_image(f"{batch_idx}/output",make_grid(xhat))