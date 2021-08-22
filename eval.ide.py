from pl_examples import _DATASETS_PATH

from models.ide import IDE
from utils import Evaluator

CKPT_PATH = fr'lightning_logs\version_4\checkpoints\epoch=100-step=1009.ckpt'


def main(args=None):
    encoder = IDE.load_from_checkpoint(CKPT_PATH).encoder
    evaluator = Evaluator(root=_DATASETS_PATH,
                          encoder=encoder, output_prefix='ide')

    for _ in range(5):
        evaluator.visualize_rand()

    evaluator.summary([1, 3, 5])

    print('done')


if __name__ == "__main__":
    main()

"""
# Accuracies:
- top-1: 0.5578978622327792
- top-3: 0.6647862232779097
- top-5: 0.7102137767220903
"""
