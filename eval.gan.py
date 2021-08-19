from pl_examples import _DATASETS_PATH

from models.gan import GAN
from utils import Evaluator

CKPT_PATH = fr'lightning_logs\version_2\checkpoints\epoch=104-step=33914.ckpt'


def main(args=None):
    encoder = GAN.load_from_checkpoint(CKPT_PATH).eval().encoder
    evaluator = Evaluator(root=_DATASETS_PATH,
                          encoder=encoder, output_prefix='gan')

    for _ in range(5):
        evaluator.visualize_rand()

    evaluator.summary([1, 3, 5, 10, 50])

    print('done')


if __name__ == "__main__":
    main()
