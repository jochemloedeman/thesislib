import pathlib
from typing import Optional

import pytorch_lightning as pl
import pytorchvideo.transforms
import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from thesislib.datasets import Kinetics
from thesislib.temporal import TemporalLabel


class Kinetics400DataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root,
            train_batch_size,
            test_batch_size,
            num_workers,
            nr_frames,
            prompt_prefix
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.nr_frames = nr_frames
        self.prompt_prefix = prompt_prefix

    def setup(self, stage: Optional[str] = None) -> None:
        root_dir = pathlib.Path(self.data_root) / 'kinetics'
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(
                lambda x: torch.permute(x, dims=(1, 0, 2, 3))
            ),
            pytorchvideo.transforms.ConvertUint8ToFloat(),
            torchvision.transforms.Resize(
                size=224, interpolation=InterpolationMode.BICUBIC,
                max_size=None, antialias=None
            ),
            torchvision.transforms.CenterCrop(size=224),
            pytorchvideo.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
            torchvision.transforms.Lambda(
                lambda x: torch.permute(x, dims=(1, 0, 2, 3))
            ),
        ])
        self.kinetics_train = Kinetics(
            root=root_dir.as_posix(),
            frames_per_vid=self.nr_frames,
            num_classes='400',
            split='train',
            transform=self.transforms
        )
        self.kinetics_test = Kinetics(
            root=root_dir.as_posix(),
            frames_per_vid=self.nr_frames,
            num_classes='400',
            split='val',
            transform=self.transforms
        )
        self.kinetics_test = Kinetics(
            root=root_dir.as_posix(),
            frames_per_vid=self.nr_frames,
            num_classes='400',
            split='test',
            transform=self.transforms
        )
        self._calculate_index_to_prompt()
        self._calculate_index_to_label()
        print()

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return DataLoader(
            dataset=self.kinetics_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def _calculate_index_to_prompt(self):
        classes = [
            class_str.replace("_", " ")
            for class_str in self.kinetics_test.classes
        ]
        self.prompts = [
            self.prompt_prefix + class_str.lower() for class_str in classes
        ]
        self.index_to_prompt = {
            idx: self.prompts[idx] for idx in range(len(self.prompts))
        }

    def _calculate_index_to_label(self):
        self.index_to_label = {
            idx: TemporalLabel.TEMPORAL
            for idx in range(len(self.kinetics_test.classes))
        }


if __name__ == '__main__':
    module = Kinetics400DataModule(
        test_batch_size=1,
        num_workers=4
    )
    module.setup()
    print()
