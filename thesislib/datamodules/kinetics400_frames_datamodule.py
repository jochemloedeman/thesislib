import json
import pathlib

import pandas
import pandas as pd
from typing import Optional

import pytorch_lightning as pl
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from thesislib.datasets import VideoFrameDataset
from thesislib.datasets.video_dataset import ImglistToTensor
from thesislib.temporal import TemporalLabel


class Kinetics400FramesDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root,
            train_batch_size,
            test_batch_size,
            num_workers,
            frames_per_vid,
            prompt_prefix
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.frames_per_vid = frames_per_vid
        self.prompt_prefix = prompt_prefix

    def setup(self, stage: Optional[str] = None) -> None:
        root_dir = pathlib.Path(self.data_root) / 'kinetics_frames'
        annotation_train_path = root_dir / 'train_processed.txt'
        annotation_val_path = root_dir / 'val_processed.txt'
        annotation_test_path = root_dir / 'test_processed.txt'
        labels_path = root_dir / 'labels_to_id.csv'

        self.label_index = self._load_index(labels_path)

        self.transforms = torchvision.transforms.Compose([
            ImglistToTensor(),
            torchvision.transforms.Resize(
                size=224, interpolation=InterpolationMode.BICUBIC,
                max_size=None, antialias=None
            ),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])
        self.kinetics_train = VideoFrameDataset(
            root_path=(root_dir / 'frames').as_posix(),
            annotationfile_path=annotation_train_path.as_posix(),
            imagefile_template='frame_{:012d}.jpg',
            num_segments=self.frames_per_vid,
            frames_per_segment=1,
            transform=self.transforms
        )
        self.kinetics_val = VideoFrameDataset(
            root_path=(root_dir / 'frames').as_posix(),
            annotationfile_path=annotation_val_path.as_posix(),
            imagefile_template='frame_{:012d}.jpg',
            num_segments=self.frames_per_vid,
            frames_per_segment=1,
            transform=self.transforms
        )
        self.kinetics_test = VideoFrameDataset(
            root_path=(root_dir / 'frames').as_posix(),
            annotationfile_path=annotation_val_path.as_posix(),
            imagefile_template='frame_{:012d}.jpg',
            num_segments=self.frames_per_vid,
            frames_per_segment=1,
            transform=self.transforms
        )
        self._calculate_index_to_prompt()
        self._calculate_index_to_label()
        print()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.kinetics_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.kinetics_val,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.kinetics_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    @staticmethod
    def _load_index(file_path):
        df = pandas.read_csv(file_path)
        index_dict = df.to_dict()['name']
        return index_dict

    def _calculate_index_to_prompt(self):
        self.index_to_prompt = {
            index: self.prompt_prefix + label.lower()
            for index, label in self.label_index.items()
        }

    def _calculate_index_to_label(self):
        self.index_to_label = {
            idx: TemporalLabel.TEMPORAL
            for idx in self.label_index.keys()
        }


if __name__ == '__main__':
    module = SSV2DataModule(
        test_batch_size=2,
        num_workers=4,
        frames_per_vid=4
    )
    module.setup()
    print()
