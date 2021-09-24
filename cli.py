from pytorch_lightning.utilities.cli import LightningCLI

from reidgan.datamodules.paired_mnist import PairedMNISTDataModule
from reidgan.models.gan import GAN
from utils import set_persistent_workers


def main():
    set_persistent_workers(PairedMNISTDataModule)
    trainer_defaults = {'gpus': -1}
    cli = LightningCLI(GAN,
                       PairedMNISTDataModule,
                       trainer_defaults=trainer_defaults,
                       seed_everything_default=0, save_config_overwrite=True)


if __name__ == "__main__":
    main()
