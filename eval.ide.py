from pl_examples import _DATASETS_PATH

from models.ide import IDE
from utils import Evaluator

CKPT_PATH = fr'lightning_logs\version_4\checkpoints\epoch=100-step=1009.ckpt'


def main(args=None):
    encoder = IDE.load_from_checkpoint(CKPT_PATH).eval().encoder
    evaluator = Evaluator(root=_DATASETS_PATH, encoder=encoder, output_prefix='ide')

    for _ in range(5):
        evaluator.visualize_rand()

    top1_acc = evaluator.evaluate_top_k_accuracy(1)
    top5_acc = evaluator.evaluate_top_k_accuracy(5)
    top10_acc = evaluator.evaluate_top_k_accuracy(10)

    print('accuracy')
    print(f' top1: {top1_acc}\n top5: {top5_acc}\n top10: {top10_acc}')

    print('done')


if __name__ == "__main__":
    main()
