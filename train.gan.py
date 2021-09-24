import pytorch_lightning as pl
from pl_examples import _DATASETS_PATH

from reidgan.callbacks import LatentDimInterpolator, LatentSpaceVisualizer
from reidgan.datamodules.market1501 import PairedMarket1501DataModule
from reidgan.models.gan import GAN


def main(args=None):
    datamodule = PairedMarket1501DataModule(_DATASETS_PATH, num_workers=4,
                                            batch_size=24, shuffle=True, drop_last=True)

    model = GAN(
        latent_related_dim=512,
        latent_unrelated_dim=128,
        img_dim=datamodule.dims,
        num_classes=751,
        enc_type='resnet18',
        lr=2e-4,
        epoch_pretraining=0,
        epoch_posttraining=100,
    )

    # model = GAN.load_from_checkpoint(fr"lightning_logs\version_2\checkpoints\epoch=99-step=32299.ckpt")
    model.hparams.epoch_pretraining = 0

    dataset = datamodule.dataset_cls(
        _DATASETS_PATH, download=False, transform=datamodule.default_transforms())

    callbacks = [
        LatentSpaceVisualizer(dataset),
        LatentDimInterpolator(dataset),
    ]

    trainer = pl.Trainer(
        gpus=-1 if datamodule.num_workers > 0 else None,
        progress_bar_refresh_rate=1,
        max_epochs=1000,
        callbacks=callbacks
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
