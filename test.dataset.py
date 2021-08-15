from datasets.market1501 import PairedMarket1501
from datamodules.market1501 import PairedMarket1501DataModule
from pl_examples import _DATASETS_PATH
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch


def main(args=None):
    dm = PairedMarket1501DataModule(
        _DATASETS_PATH, batch_size=1, num_workers=0)
    dm.setup()
    ds = dm.train_dataloader()
    it = iter(ds)

    imgs = []
    for _ in range(5):
        batch = next(it)
        x, y = batch
        imgs.append(x[0])
        imgs.append(x[1])
        imgs.append(x[2])

    grid = make_grid(torch.cat(imgs, 0), 3)
    print('done')


if __name__ == "__main__":
    main()
