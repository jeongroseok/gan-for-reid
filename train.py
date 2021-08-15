import pytorch_lightning as pl
from pl_examples import _DATASETS_PATH

from callbacks import LatentDimInterpolator, LatentSpaceVisualizer
from datamodules.market1501 import PairedMarket1501DataModule
from models.gan import GAN


def main(args=None):
    datamodule = PairedMarket1501DataModule(_DATASETS_PATH, num_workers=6,
                                      batch_size=32, shuffle=True, drop_last=True)

    model = GAN(64, 64, datamodule.dims, lr=2e-4,
                normalize=True, hidden_dim=1024, noise_dim=16, epoch_pretraining=50, num_classes=751)

    dataset = datamodule.dataset_cls(
        _DATASETS_PATH, download=False, transform=datamodule.default_transforms())

    callbacks = [
        LatentSpaceVisualizer(dataset),
        LatentDimInterpolator(dataset),
    ]

    trainer = pl.Trainer(
        gpus=-1,
        progress_bar_refresh_rate=1,
        max_epochs=1000,
        callbacks=callbacks
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
