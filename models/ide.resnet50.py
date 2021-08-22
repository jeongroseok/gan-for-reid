import pytorch_lightning as pl
import torch
import torchmetrics.classification.accuracy

from torchvision.models.resnet import resnet50
from .components import *


class IDE(pl.LightningModule):
    def __init__(
            self,
            img_dim: tuple[int, int, int] = (1, 28, 28),
            num_classes: int = 10,
            lr: float = 1e-4,
            adam_beta1: float = 0.5,
            *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.img_dim = img_dim
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.metric_accuracy = torchmetrics.classification.accuracy.Accuracy()

        self.resnet50 = resnet50(False, num_classes=num_classes)

    def forward(self, x):
        y_hat = self.resnet50.forward(x)
        return y_hat

    def configure_optimizers(self):
        lr = self.hparams.lr
        beta1 = self.hparams.adam_beta1
        betas = (beta1, 0.999)

        return torch.optim.Adam(self.parameters(), lr, betas)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion_cls(y_hat, y)
        accuracy = self.metric_accuracy(y_hat, y)
        self.log(f"{self.__class__.__name__}/loss", loss)
        self.log(f"{self.__class__.__name__}/accuracy", accuracy)
        return loss
