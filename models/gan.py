import pytorch_lightning as pl
import torch
import torchmetrics.classification.accuracy
from torch import ones_like, zeros_like
from torch.nn import functional as F

from .components import *


def kl_loss(p, q, z):
    log_qz = q.log_prob(z)
    log_pz = p.log_prob(z)
    kl = log_qz - log_pz
    kl = kl.mean()
    return kl


class GAN(pl.LightningModule):
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
        self.criterion_recon = torch.nn.L1Loss()
        self.criterion_adv = torch.nn.BCELoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_kld = kl_loss
        self.metric_accuracy = torchmetrics.classification.accuracy.Accuracy()

        self.encoder = Encoder(latent_related_dim, latent_unrelated_dim, img_dim,
                               num_classes, hidden_dim, normalize)
        self.decoder = Decoder(latent_related_dim, latent_unrelated_dim, img_dim,
                               hidden_dim, normalize, noise_dim)
        self.discriminator = Discriminator(
            img_dim, num_classes, hidden_dim, normalize)

        self.automatic_optimization = False

    def encode(self, x):
        p, q, z_related, z_unrelated, y_hat = self.encoder.forward(x)
        return p, q, z_related, z_unrelated, y_hat

    def decode(self, z_related, z_unrelated):
        x_hat = self.decoder.forward(z_related, z_unrelated)
        return x_hat

    def discriminate(self, x):
        d, y_hat = self.discriminator.forward(x)
        return d, y_hat

    def forward(self, z_related, z_unrelated):
        return self.decode(z_related, z_unrelated)

    def configure_optimizers(self):
        lr = self.hparams.lr
        beta1 = self.hparams.adam_beta1
        betas = (beta1, 0.999)

        parameters_enc = \
            list(self.encoder.classifier.parameters()) + \
            list(self.encoder.backbone.parameters()) + \
            list(self.encoder.fc_related.parameters()) + \
            list(self.encoder.fc_unrelated.parameters())

        parameters_gen = \
            list(self.decoder.parameters()) + \
            list(self.encoder.backbone.parameters()) + \
            list(self.encoder.fc_related.parameters()) + \
            list(self.encoder.fc_unrelated.parameters())

        parameters_disc = self.discriminator.parameters()

        opt_enc = torch.optim.Adam(parameters_enc, lr, betas)
        opt_gen = torch.optim.Adam(parameters_gen, lr, betas)
        opt_disc = torch.optim.Adam(parameters_disc, lr, betas)

        return [opt_enc, opt_gen, opt_disc]

    def training_step(self, batch, batch_idx):
        opt_e, opt_g, opt_d = self.optimizers()

        (x_anchor, x_positive, x_negative), (y_anchor, y_positive, y_negative) = batch

        if self.current_epoch < self.hparams.epoch_pretraining:
            pass
            # encoder
            loss = self._encoder_step(x_anchor, y_anchor)
            opt_e.zero_grad()
            self.manual_backward(loss)
            opt_e.step()
        else:
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
        _, _, _, _, y_hat = self.encode(x)
        loss = self.criterion_cls(y_hat, y)
        accuracy = self.metric_accuracy(y_hat, y)
        self.log(f"{self.__class__.__name__}/encoder/accuracy", accuracy)
        return loss

    def _discriminator_step(self, x_a, x_p, y):
        # Real
        d_real, y_real = self.discriminate(x_a)
        loss_adv_real = self.criterion_adv(d_real, ones_like(d_real))
        loss_cls_real = self.criterion_cls(y_real, y)
        loss_real = loss_adv_real + (0.5 * loss_cls_real)

        # Same
        p_a, q_a, z_rel_a, z_unrel_a, y_hat_a = self.encode(x_a)
        x_hat_same = self.decode(z_rel_a, z_unrel_a)
        d_fake_same, y_fake_same = self.discriminate(x_hat_same)
        loss_adv_same = self.criterion_adv(
            d_fake_same, zeros_like(d_fake_same))
        loss_cls_same = self.criterion_cls(y_fake_same, y)
        loss_same = loss_adv_same + (0.5 * loss_cls_same)

        # Diff
        p_p, q_p, z_rel_p, z_unrel_p, y_hat_p = self.encode(x_p)
        x_hat_diff = self.decode(z_rel_a, z_unrel_p)
        d_fake_diff, y_fake_diff = self.discriminate(x_hat_diff)
        loss_adv_diff = self.criterion_adv(
            d_fake_diff, zeros_like(d_fake_diff))
        loss_cls_diff = self.criterion_cls(y_fake_diff, y)
        loss_diff = loss_adv_diff + (0.5 * loss_cls_diff)

        # Logging
        self.log(f"{self.__class__.__name__}/discriminator/adv",
                 (loss_adv_real + loss_adv_same + loss_adv_diff) / 3)

        accuracy_real = self.metric_accuracy(y_real, y)
        accuracy_same = self.metric_accuracy(y_fake_same, y)
        accuracy_diff = self.metric_accuracy(y_fake_diff, y)
        self.log(f"{self.__class__.__name__}/discriminator/accuracy",
                 (accuracy_real + accuracy_same + accuracy_diff) / 3)
        return loss_real + loss_same + loss_diff

    def _generator_step(self, x_a, x_p, y):
        # Same
        p_a, q_a, z_rel_a, z_unrel_a, y_hat_a = self.encode(x_a)
        x_hat_same = self.decode(z_rel_a, z_unrel_a)
        d_fake_same, y_fake_same = self.discriminate(x_hat_same)
        loss_adv_same = self.criterion_adv(
            d_fake_same, ones_like(d_fake_same))
        loss_cls_same = self.criterion_cls(y_fake_same, y)
        loss_recon_same = self.criterion_recon(x_hat_same, x_a)
        loss_kld_same = self.criterion_kld(p_a, q_a, z_unrel_a)

        loss_same = \
            loss_adv_same + \
            (2 * loss_cls_same) + \
            (10 * loss_recon_same) + \
            (1 * loss_kld_same)

        # Diff
        p_p, q_p, z_rel_p, z_unrel_p, y_hat_p = self.encode(x_p)
        x_hat_diff = self.decode(z_rel_a, z_unrel_p)
        d_fake_diff, y_fake_diff = self.discriminate(x_hat_diff)
        loss_adv_diff = self.criterion_adv(
            d_fake_diff, ones_like(d_fake_diff))
        loss_cls_diff = self.criterion_cls(y_fake_diff, y)
        loss_recon_diff = self.criterion_recon(x_hat_diff, x_p)
        loss_kld_diff = self.criterion_kld(p_p, q_p, z_unrel_p)

        loss_diff = \
            loss_adv_diff + \
            (2 * loss_cls_diff) + \
            (10 * loss_recon_diff) + \
            (1 * loss_kld_diff)

        # Logging
        self.log(f"{self.__class__.__name__}/generator/adv",
                 (loss_adv_same + loss_adv_diff) / 2)
        self.log(f"{self.__class__.__name__}/generator/recon",
                 (loss_recon_same + loss_recon_diff) / 2)
        self.log(f"{self.__class__.__name__}/generator/kld",
                 (loss_kld_same + loss_kld_diff) / 2)
        accuracy_same = self.metric_accuracy(y_fake_same, y)
        accuracy_diff = self.metric_accuracy(y_fake_diff, y)
        self.log(f"{self.__class__.__name__}/generator/accuracy",
                 (accuracy_same + accuracy_diff) / 2)

        return loss_same + loss_diff
