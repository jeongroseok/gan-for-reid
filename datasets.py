from typing import Optional
import numpy as np
from PIL import Image

from torch.utils.data.sampler import BatchSampler
from torchvision.datasets import MNIST


class PairedMNIST(MNIST):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[callable] = None,
            target_transform: Optional[callable] = None,
            download: bool = False,
    ):
        PairedMNIST.__name__ = MNIST.__name__ # optim trick
        super().__init__(root, train, transform, target_transform, download)

        self.targets_set = set(self.targets.numpy())
        self.target_to_indices = {target: np.where(self.targets.numpy() == target)[
            0] for target in self.targets_set}

    def __getitem__(self, index_a):
        img_a, target_a = self.data[index_a], int(self.targets[index_a])

        index_p = index_a
        while index_p == index_a:
            index_p = np.random.choice(self.target_to_indices[target_a])
        img_p, target_p = self.data[index_p], int(self.targets[index_p])

        target_n = np.random.choice(list(self.targets_set - set([target_a])))
        index_n = np.random.choice(self.target_to_indices[target_n])
        img_n, target_n = self.data[index_n], int(self.targets[index_n])

        img_a = Image.fromarray(img_a.numpy(), mode='L')
        img_p = Image.fromarray(img_p.numpy(), mode='L')
        img_n = Image.fromarray(img_n.numpy(), mode='L')

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)

        if self.target_transform is not None:
            target_a = self.target_transform(target_a)
            target_p = self.target_transform(target_p)
            target_n = self.target_transform(target_n)

        return (img_a, img_p, img_n), (target_a, target_p, target_n)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(
                self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                   class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
