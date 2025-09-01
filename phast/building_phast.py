from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import instantiate_class

class BuildingPHAST(LightningModule):
    def __init__(
        self,
        backbone: nn.Module, 
        decoder: nn.Module,
        criterion: nn.Module,
        metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        optimizer: Optional[dict] = None,
        lr_scheduler: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.criterion = criterion
        self.metrics = nn.ModuleDict(metrics or {})
        self.opt_cfg = optimizer
        self.sch_cfg = lr_scheduler
        self.save_hyperparameters(ignore=["backbone", "decoder", "criterion", "metrics"])

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.decoder(feats)
        return logits

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        for k, m in self.metrics.items():
            self.log(f"{stage}/{k}", m(logits, y), prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, _):  
        return self._step(batch, "train")
    
    def validation_step(self, batch, _): 
        self._step(batch, "val")

    def test_step(self, batch, _):       
        self._step(batch, "test")

    def configure_optimizers(self):
        if self.trainer is not None and hasattr(self.trainer, "strategy"):
            return

