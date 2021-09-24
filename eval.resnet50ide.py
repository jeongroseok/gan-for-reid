from pl_examples import _DATASETS_PATH

from reidgan.models.ide import ResNet50IDE
from reidgan.utils import Evaluator

CKPT_PATH = fr'lightning_logs\version_0\checkpoints\epoch=26-step=701.ckpt'


def main(args=None):
    encoder = ResNet50IDE.load_from_checkpoint(CKPT_PATH)
    evaluator = Evaluator(root=_DATASETS_PATH,
                          encoder=encoder, output_prefix='res50ide', batch_size=1024)

    for _ in range(5):
        evaluator.visualize_rand()

    evaluator.summary([1, 3, 5])

    print('done')


if __name__ == "__main__":
    main()
