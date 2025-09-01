import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.cli import LightningCLI

if __name__ == "__main__":
    LightningCLI(
        
        subclass_mode_model=True, subclass_mode_data=True,
        auto_configure_optimizers=True,
    )