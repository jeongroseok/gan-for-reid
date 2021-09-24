from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics.classification.accuracy
from torch import nn, ones_like, zeros_like

from .components.basic import *
from .components.resnet import *


class GAN(pl.LightningModule):
    class __HPARAMS:
        lr: float
        adam_beta1: float
        epoch_pretraining: int
        epoch_training: int
        epoch_posttraining: int

    hparams: __HPARAMS
    encoder: ResNetEncoder
    decoder: ResNetDecoder

    def __init__(
        self,
        latent_related_dim: int = 32,
        latent_unrelated_dim: int = 32,
        img_dim: Tuple[int, int, int] = (1, 28, 28),
        num_classes: int = 10,
        enc_type: str = "resnet18",
        lr: float = 1e-4,
        adam_beta1: float = 0.5,
        epoch_pretraining: int = 100,
        epoch_training: int = 100,
        epoch_posttraining: int = 0,
        first_conv: bool = False,
        maxpool1: bool = False,
        *args: any,
        **kwargs: any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.criterion_recon = torch.nn.L1Loss()
        self.criterion_adv = torch.nn.BCELoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_kld = nn.KLDivLoss(reduction="batchmean")

        self.metric_accuracy = torchmetrics.classification.accuracy.Accuracy()

        self.encoder = create_encoder(
            enc_type,
            latent_related_dim,
            latent_unrelated_dim,
            img_dim,
            first_conv,
            maxpool1,
        )
        self.decoder = create_decoder(
            enc_type,
            latent_related_dim,
            latent_unrelated_dim,
            img_dim,
            first_conv,
            maxpool1,
        )
        self.encoder_guider = BasicClassifier(num_classes, latent_related_dim, 128)
        self.discriminator = BasicDiscriminator(img_dim, num_classes)

    def forward(self, z_related, z_unrelated):
        return self.decoder.forward(
            z_related, z_unrelated
        )  # 랜덤 z 넣어줘야함. 안하면 출력값이 deterministic

    def configure_optimizers(self):
        lr = self.hparams.lr
        beta1 = self.hparams.adam_beta1
        betas = (beta1, 0.999)

        parameters_enc = list(self.encoder.parameters()) + list(
            self.encoder_guider.parameters()
        )

        parameters_gen = list(self.encoder.parameters()) + list(
            self.decoder.parameters()
        )

        parameters_disc = self.discriminator.parameters()

        opt_enc = torch.optim.Adam(parameters_enc, lr, betas)
        opt_gen = torch.optim.Adam(parameters_gen, lr, betas)
        opt_disc = torch.optim.Adam(parameters_disc, lr, betas)

        return [opt_enc, opt_gen, opt_disc]

    def training_step(self, batch, batch_idx):
        opt_e, opt_g, opt_d = self.optimizers()

        (x_anchor, x_positive, x_negative), (y_anchor, y_positive, y_negative) = batch

        if self.current_epoch < self.hparams.epoch_pretraining:
            # encoder
            loss = self._encoder_step(x_anchor, y_anchor)
            opt_e.zero_grad()
            self.manual_backward(loss)
            opt_e.step()
        elif self.current_epoch >= self.hparams.epoch_pretraining:
            # discriminator
            loss = self._discriminator_step(x_anchor, x_positive, y_anchor)
            opt_d.zero_grad()
            self.manual_backward(loss)
            opt_d.step()
            # generator
            loss = self._generator_step(x_anchor, x_positive, y_anchor)
            opt_g.zero_grad()
            self.manual_backward(loss)
            opt_g.step()
        elif (
            self.current_epoch
            >= self.hparams.epoch_pretraining + self.hparams.epoch_training
        ):
            # fine-tuning
            loss = self._encoder_step(x_anchor, y_anchor)
            opt_e.zero_grad()
            self.manual_backward(loss)
            opt_e.step()
            # discriminator
            loss = self._discriminator_step(x_anchor, x_positive, y_anchor)
            opt_d.zero_grad()
            self.manual_backward(loss)
            opt_d.step()
            # generator
            loss = self._generator_step(x_anchor, x_positive, y_anchor)
            opt_g.zero_grad()
            self.manual_backward(loss)
            opt_g.step()

    def _encoder_step(self, x, y):
        z_related, z_unrelated = self.encoder.forward(x)
        y_hat = self.encoder_guider.forward(z_related)
        loss = self.criterion_cls(y_hat, y)
        accuracy = self.metric_accuracy(y_hat, y)
        self.log(f"{self.__class__.__name__}/encoder_guider/accuracy", accuracy)
        return loss

    def _discriminator_step(self, x_anchor, x_positive, y):
        def _loss(y, y_hat, d, real: bool):
            adv = self.criterion_adv(d, ones_like(d) if real else zeros_like(d))
            cls = self.criterion_cls(y_hat, y)
            return adv, cls

        def _forward(z_related, z_unrelated):
            x_hat = self.decoder.forward(z_related, z_unrelated)
            d, y_hat = self.discriminator.forward(x_hat)
            return y_hat, d

        # Real
        d_real, y_real = self.discriminator.forward(x_anchor)
        loss_adv_real, loss_cls_real = _loss(y, y_real, d_real, True)
        loss_real = loss_adv_real + (0.5 * loss_cls_real)

        # Fake
        z_related_anchor, z_unrelated_anchor = self.encoder.forward(x_anchor)
        z_related_positive, z_unrealted_positive = self.encoder.forward(x_positive)

        # - Same
        y_hat_same, d_same = _forward(z_related_anchor, z_unrelated_anchor)
        loss_adv_same, loss_cls_same = _loss(y, y_hat_same, d_same, False)
        loss_same = loss_adv_same + (0.5 * loss_cls_same)

        # - Diff
        y_hat_diff, d_diff = _forward(z_related_anchor, z_unrealted_positive)
        loss_adv_diff, loss_cls_diff = _loss(y, y_hat_diff, d_diff, False)
        loss_diff = loss_adv_diff + (0.5 * loss_cls_diff)

        # Logging
        self.log(
            f"{self.__class__.__name__}/discriminator/adv",
            (loss_adv_real + loss_adv_same + loss_adv_diff) / 3,
        )

        accuracy_real = self.metric_accuracy(y_real, y)
        accuracy_same = self.metric_accuracy(y_hat_same, y)
        accuracy_diff = self.metric_accuracy(y_hat_diff, y)
        self.log(
            f"{self.__class__.__name__}/discriminator/accuracy",
            (accuracy_real + accuracy_same + accuracy_diff) / 3,
        )
        return loss_real + loss_same + loss_diff

    def _generator_step(self, x_anchor, x_positive, y):
        def _forward(z_related, z_unrelated):
            x_hat = self.decoder.forward(z_related, z_unrelated)
            d, y_hat = self.discriminator.forward(x_hat)
            return x_hat, y_hat, d

        def _loss(z_unrelated, x, x_hat, y, y_hat, d):
            adv = self.criterion_adv(d, ones_like(d))
            cls = self.criterion_cls(y_hat, y)
            recon = self.criterion_recon(x_hat, x)
            kl = (
                self.criterion_kld(
                    F.log_softmax(z_unrelated, 1), torch.ones_like(z_unrelated)
                )
                * 0.1
            )
            return adv, cls, recon, kl

        # Same
        z_related_anchor, z_unrelated_anchor = self.encoder.forward(x_anchor)
        x_hat_same, y_hat_same, d_same = _forward(z_related_anchor, z_unrelated_anchor)
        loss_adv_same, loss_cls_same, loss_recon_same, loss_kld_same = _loss(
            z_unrelated_anchor, x_anchor, x_hat_same, y, y_hat_same, d_same
        )
        loss_same = (
            loss_adv_same
            + (2 * loss_cls_same)
            + (10 * loss_recon_same)
            + (0.1 * loss_kld_same)
        )

        # Diff
        z_related_positive, z_unrelated_positive = self.encoder.forward(x_positive)
        x_hat_diff, y_hat_diff, d_diff = _forward(
            z_related_anchor, z_unrelated_positive
        )
        loss_adv_diff, loss_cls_diff, loss_recon_diff, loss_kld_diff = _loss(
            z_unrelated_positive, x_positive, x_hat_diff, y, y_hat_diff, d_diff
        )
        loss_diff = (
            loss_adv_diff
            + (2 * loss_cls_diff)
            + (10 * loss_recon_diff)
            + (0.1 * loss_kld_diff)
        )

        # Logging
        self.log(
            f"{self.__class__.__name__}/generator/adv",
            (loss_adv_same + loss_adv_diff) / 2,
        )
        self.log(
            f"{self.__class__.__name__}/generator/recon",
            (loss_recon_same + loss_recon_diff) / 2,
        )
        self.log(
            f"{self.__class__.__name__}/generator/kld",
            (loss_kld_same + loss_kld_diff) / 2,
        )
        accuracy_same = self.metric_accuracy(y_hat_same, y)
        accuracy_diff = self.metric_accuracy(y_hat_diff, y)
        self.log(
            f"{self.__class__.__name__}/generator/accuracy",
            (accuracy_same + accuracy_diff) / 2,
        )

        return loss_same + loss_diff
