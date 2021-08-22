import pytorch_lightning as pl
from pl_examples import _DATASETS_PATH

from callbacks import LatentDimInterpolator, LatentSpaceVisualizer
from datamodules.market1501 import Market1501DataModule
from models.ide import ResNet50IDE


def main(args=None):
    datamodule = Market1501DataModule(_DATASETS_PATH, num_workers=4,
                                      batch_size=384, shuffle=True, drop_last=True)

    model = ResNet50IDE(datamodule.dims, num_classes=751, lr=1e-3)

    trainer = pl.Trainer(
        gpus=-1 if datamodule.num_workers > 0 else None,
        progress_bar_refresh_rate=1,
        max_epochs=300,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
