import pytorch_lightning as pl
import torch
import torchmetrics.classification.accuracy

from .components import *


class IDE(pl.LightningModule):
    def __init__(
            self,
            latent_related_dim: int = 32,
            latent_unrelated_dim: int = 32,
            img_dim: tuple[int, int, int] = (1, 28, 28),
            num_classes: int = 10,
            lr: float = 1e-4,
            adam_beta1: float = 0.5,
            hidden_dim: int = 256,
            normalize: bool = True,
            noise_dim: int = None,
            epoch_pretraining: int = 5,
            *args: any, **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.img_dim = img_dim
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.metric_accuracy = torchmetrics.classification.accuracy.Accuracy()

        self.encoder = Encoder(latent_related_dim, latent_unrelated_dim, img_dim,
                               num_classes, hidden_dim, normalize)

    def forward(self, x):
        _, _, _, _, y_hat = self.encoder.forward(x)
        return y_hat

    def configure_optimizers(self):
        lr = self.hparams.lr
        beta1 = self.hparams.adam_beta1
        betas = (beta1, 0.999)

        parameters_enc = \
            list(self.encoder.classifier.parameters()) + \
            list(self.encoder.backbone.parameters()) + \
            list(self.encoder.fc_related.parameters())

        return torch.optim.Adam(parameters_enc, lr, betas)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion_cls(y_hat, y)
        accuracy = self.metric_accuracy(y_hat, y)
        self.log(f"{self.__class__.__name__}/loss", loss)
        self.log(f"{self.__class__.__name__}/accuracy", accuracy)
        return loss
