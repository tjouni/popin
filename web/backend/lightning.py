import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from networks import Regressor, Pretrained_regressor


class Insta(pl.LightningModule):
    def __init__(self, dim=32, lr=3e-4, weight_decay=1e-3, transform_target=False, pretrained=False):
        super().__init__()
        
        self.save_hyperparameters()
        
        if self.hparams.pretrained:
            self.net = Pretrained_regressor()
        else:
            self.net = Regressor(dim)
        self.loss = nn.L1Loss()
        
    def forward(self, x):
        """
        A forward function to be used during inference.
        
        Args:
            x (Tensor): image to be scored.
        """
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        
        scores = self.net(imgs)
        loss = self.loss(scores, targets)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        
        scores = self.net(imgs)
        loss = self.loss(scores, targets)
        
        self.log('val/loss', loss, on_step=True, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer