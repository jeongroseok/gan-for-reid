import pytorch_lightning as pl
from pl_examples import _DATASETS_PATH

from reidgan.callbacks import LatentDimInterpolator, LatentSpaceVisualizer
from reidgan.datamodules.market1501 import Market1501DataModule
from reidgan.models.ide import IDE


def main(args=None):
    datamodule = Market1501DataModule(_DATASETS_PATH, num_workers=0,
                                      batch_size=1024, shuffle=True, drop_last=True)

    model = IDE(256, 192, datamodule.dims, lr=2e-4,
                normalize=True, hidden_dim=2048, noise_dim=16, epoch_pretraining=50, num_classes=751)

    trainer = pl.Trainer(
        gpus=-1 if datamodule.num_workers > 0 else None,
        progress_bar_refresh_rate=1,
        max_epochs=300,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
