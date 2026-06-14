from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from dataset.data_helper import create_pretrain_datasets

class Pretrain_DataModule(LightningDataModule):

    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args

    def setup(self, stage: str):
        pretrain_dataset = create_pretrain_datasets(self.args)
        self.dataset = {
               "train": pretrain_dataset
        }
    def train_dataloader(self):
        loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=True,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader
