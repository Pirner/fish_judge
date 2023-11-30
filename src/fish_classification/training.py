import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy


class FishClassificationModule(pl.LightningModule):
    def __init__(self, model, n_classes: int):
        super(FishClassificationModule, self).__init__()
        self.model = model
        self.criterion = nn.BCELoss()

        self.test_preds = []
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, sample):
        x = sample
        logit = self.model(x)
        return logit

    def training_step(self, batch, batch_idx):
        sample, y = batch

        preds = self(sample)
        loss = self.criterion(preds, y)
        acc = self.accuracy(preds, torch.argmax(y, dim=-1))
        self.log('train_loss', loss.item(), on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sample, y = batch

        preds = self(sample)

        loss = self.criterion(preds, y)
        acc = self.accuracy(preds, torch.argmax(y, dim=-1))

        self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', loss.item(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer

    def test_step(self, batch, batch_idx):
        sample = batch
        preds = 100 * self(sample)
        self.test_preds.append(preds.detach().cpu())

    def get_predictions(self):
        return torch.cat(self.test_preds).numpy()
