import pathlib
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import pytorchvideo.transforms
import torch
import torchvision.transforms
from pytorchvideo.data import Kinetics
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from thesislib.temporal import TemporalLabel


class Kinetics400DataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root,
            train_batch_size,
            test_batch_size,
            num_workers,
            nr_frames,
            prompt_prefix,
            fps,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.nr_frames = nr_frames
        self.prompt_prefix = prompt_prefix
        self.fps = fps

    def setup(self, stage: Optional[str] = None) -> None:
        root_dir = pathlib.Path(self.data_root) / 'kinetics'
        labels_to_id = root_dir / 'annotations' / 'labels_to_id.csv'
        self.id_to_label = pd.read_csv(labels_to_id).to_dict()['name']

        self.train_transforms = pytorchvideo.transforms.ApplyTransformToKey(
            key='video',
            transform=torchvision.transforms.Compose([
                pytorchvideo.transforms.UniformTemporalSubsample(num_samples
                                                                 =self.nr_frames),
                pytorchvideo.transforms.Div255(),
                pytorchvideo.transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                ),
                pytorchvideo.transforms.ShortSideScale(size=256),
                torchvision.transforms.RandomCrop(size=224)
            ])
        )
        self.test_transforms = pytorchvideo.transforms.ApplyTransformToKey(
            key='video',
            transform=torchvision.transforms.Compose([
                pytorchvideo.transforms.UniformTemporalSubsample(num_samples
                                                                 =self.nr_frames),
                pytorchvideo.transforms.Div255(),
                pytorchvideo.transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                ),
                pytorchvideo.transforms.ShortSideScale(size=224),
                torchvision.transforms.CenterCrop(size=224)
            ])
        )
        self.kinetics_train = Kinetics(
            data_path=(
                        root_dir / 'annotations' / 'train_new.csv').as_posix(),
            clip_sampler=pytorchvideo.data.RandomClipSampler(
                clip_duration=float(self.nr_frames / self.fps)
            ),
            video_path_prefix=(root_dir / 'train_24').as_posix(),
            decode_audio=False,
            transform=self.train_transforms
        )
        self.kinetics_val = Kinetics(
            data_path=(
                        root_dir / 'annotations' / 'validate_new.csv').as_posix(),
            clip_sampler=pytorchvideo.data.RandomClipSampler(
                clip_duration=float(self.nr_frames / self.fps)
            ),
            video_path_prefix=(root_dir / 'val_24').as_posix(),
            decode_audio=False,
            transform=self.test_transforms
        )

        self.kinetics_test = Kinetics(
            data_path=(root_dir / 'annotations' / 'test_new.csv').as_posix(),
            clip_sampler=pytorchvideo.data.UniformClipSampler(
                clip_duration=float(self.nr_frames / self.fps)
            ),
            video_path_prefix=(root_dir / 'test_24').as_posix(),
            decode_audio=False,
            transform=self.test_transforms
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

    def _calculate_index_to_prompt(self):
        classes = [
            class_str.replace("_", " ")
            for class_str in self.id_to_label.values()
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
            for idx in range(len(self.id_to_label.keys()))
        }


if __name__ == '__main__':
    module = Kinetics400DataModule(
        test_batch_size=1,
        num_workers=4
    )
    module.setup()
    print()
