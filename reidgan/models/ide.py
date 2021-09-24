from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torchmetrics.classification.accuracy
from torchvision.models.resnet import resnet50

from .components.resnet import create_encoder


class ResNet50IDE(pl.LightningModule):
    class __HPARAMS:
        img_dim: Tuple[int, int, int]
        num_classes: int
        lr: float
        adam_beta1: float

    hparams: __HPARAMS

    def __init__(
        self,
        img_dim: Tuple[int, int, int] = (1, 28, 28),
        num_classes: int = 10,
        lr: float = 1e-4,
        adam_beta1: float = 0.5,
        *args: any,
        **kwargs: any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.img_dim = img_dim
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.metric_accuracy = torchmetrics.classification.accuracy.Accuracy()

        self.resnet50 = resnet50(False, num_classes=num_classes)

    def encode(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)

        return x

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


class IDE(pl.LightningModule):
    def __init__(
        self,
        enc_type: str = "resnet18",
        img_dim: Tuple[int, int, int] = (1, 28, 28),
        num_classes: int = 10,
        lr: float = 1e-4,
        adam_beta1: float = 0.5,
        *args: any,
        **kwargs: any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.img_dim = img_dim
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.metric_accuracy = torchmetrics.classification.accuracy.Accuracy()

        self.encoder = create_encoder(
            enc_type, num_classes, img_dim, False, False, False
        )

    def forward(self, x):
        y_hat = self.encoder.forward(x)
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
