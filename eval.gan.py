from pl_examples import _DATASETS_PATH

from reidgan.models.gan import GAN
from reidgan.utils import Evaluator

CKPT_PATH = fr'lightning_logs\version_3\checkpoints\epoch=41-step=18101.ckpt'


def main(args=None):
    encoder = GAN.load_from_checkpoint(CKPT_PATH).encoder
    evaluator = Evaluator(root=_DATASETS_PATH,
                          encoder=encoder, output_prefix='gan')

    for _ in range(5):
        evaluator.visualize_rand()

    evaluator.summary([1, 3, 5, 10])

    print('done')


if __name__ == "__main__":
    main()

"""
# Accuracies:
- top-1: 0.9269596199524941
- top-3: 0.9682304038004751
- top-5: 0.9798099762470309
- top-10: 0.9878266033254157
"""
