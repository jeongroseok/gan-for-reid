from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from torch.utils.data import DataLoader, Dataset


def set_persistent_workers(data_module: VisionDataModule):
    def _data_loader(self: VisionDataModule,
                     dataset: Dataset,
                     shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
    data_module._data_loader = _data_loader
