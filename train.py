import pytorch_lightning as pl
from pl_examples import _DATASETS_PATH

from callbacks import LatentDimInterpolator, LatentSpaceVisualizer
from datamodules import PairedMNISTDataModule
from models.gan import GAN
from utils import set_persistent_workers


def main(args=None):
    set_persistent_workers(PairedMNISTDataModule)

    datamodule = PairedMNISTDataModule(_DATASETS_PATH, num_workers=6,
                                       batch_size=128, shuffle=True, drop_last=True)

    model = GAN(32, 32, datamodule.dims, lr=2e-4,
                normalize=True, hidden_dim=1024, noise_dim=16, epoch_pretraining=15)

    dataset = datamodule.dataset_cls(
        _DATASETS_PATH, False, transform=datamodule.default_transforms())

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
