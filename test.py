import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision.utils
from pl_examples import _DATASETS_PATH
from torchvision import transforms as transform_lib

from datasets import PairedMNIST
import models.dgan

CKPT = fr'lightning_logs\version_15\checkpoints\epoch=48-step=18374 copy.ckpt'
CKPT = fr'lightning_logs\version_18\checkpoints\epoch=158-step=59624.ckpt'


def vis(model: models.dgan.DGAN, x: torch.Tensor):
    _, _, z, _ = model.encode(x.unsqueeze(0))
    img = model.decode(z)
    img = torch.cat((x.unsqueeze(0), img), 0)
    img = torchvision.utils.make_grid(img, normalize=True)
    plt.imshow(img.permute(1, 2, 0).detach().numpy())
    plt.show()


def main(args=None):
    transforms = transform_lib.Compose([
        transform_lib.ToTensor(),
        transform_lib.Normalize((0.5, ), (0.5, )),
    ])
    dataset = PairedMNIST(_DATASETS_PATH, False, transforms)
    (x_anchor, x_positive, x_negative), (y_anchor,
                                         y_positive, y_negative) = dataset[0]
    model = models.dgan.DGAN.load_from_checkpoint(CKPT).eval()

    vis(model, x_anchor)
    vis(model, x_positive)
    vis(model, x_negative)

    _, _, z_a, _ = model.encode(x_anchor.unsqueeze(0))
    _, _, z_p, _ = model.encode(x_positive.unsqueeze(0))
    z_1 = torch.cat([z_a[..., :128], z_p[..., 128:]], -1)
    z_2 = torch.cat([z_a[..., 128:], z_p[..., :128]], -1)
    img_1 = model.decode(z_1)[0]
    img_2 = model.decode(z_2)[0]
    plt.imshow(img_1.permute(1, 2, 0).detach().numpy())
    plt.show()
    plt.imshow(img_2.permute(1, 2, 0).detach().numpy())
    plt.show()

    print('done')


if __name__ == "__main__":
    main()
