import json
import os
import pathlib
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import pytorchvideo.transforms
import torch
import torchvision.transforms
from pytorchvideo.data import Kinetics, Hmdb51
from torch.utils.data import DataLoader

from thesislib.temporal import TemporalLabel


class HMDB51DataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root,
            data_split,
            train_batch_size,
            test_batch_size,
            num_workers,
            nr_frames,
            prompt_prefix,
            fps,
            temporal_dataset,
    ):
        super().__init__()
        self.data_root = data_root
        self.data_split = data_split
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.nr_frames = nr_frames
        self.prompt_prefix = prompt_prefix
        self.fps = fps
        self.temporal_dataset = temporal_dataset

    def setup(self, stage: Optional[str] = None) -> None:
        root_dir = pathlib.Path(self.data_root) / 'hmdb51'
        labels_to_id_path = root_dir / 'labels_to_id.csv'
        class_to_prompt_path = root_dir / 'class_to_prompt.json'
        self.id_to_class = pd.read_csv(labels_to_id_path).to_dict()['name']
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}
        with open(class_to_prompt_path) as file:
            self.class_to_prompt = json.load(file)

        self.train_transform = pytorchvideo.transforms.ApplyTransformToKey(
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
        self.test_transform = pytorchvideo.transforms.ApplyTransformToKey(
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
        if stage == 'fit':
            self.hmdb51_train = Hmdb51(
                data_path=(root_dir / 'splits'),
                clip_sampler=pytorchvideo.data.RandomClipSampler(
                    clip_duration=float(self.nr_frames / self.fps)
                ),
                video_sampler=torch.utils.data.DistributedSampler,
                video_path_prefix=(root_dir / 'videos').as_posix(),
                decode_audio=False,
                split_type='train',
                split_id=self.data_split,
                transform=self.train_transform
            )
            self.hmdb51_val = Hmdb51(
                data_path=(root_dir / 'splits'),
                clip_sampler=pytorchvideo.data.RandomClipSampler(
                    clip_duration=float(self.nr_frames / self.fps)
                ),
                video_sampler=torch.utils.data.DistributedSampler,
                video_path_prefix=(root_dir / 'videos').as_posix(),
                decode_audio=False,
                split_type='train',
                split_id=self.data_split,
                transform=self.test_transform
            )
        if stage == 'test':
            self.hmdb51_test = Hmdb51(
                data_path=(root_dir / 'splits'),
                clip_sampler=pytorchvideo.data.UniformClipSampler(
                    clip_duration=float(self.nr_frames / self.fps)
                ),
                video_sampler=torch.utils.data.SequentialSampler,
                video_path_prefix=(root_dir / 'videos').as_posix(),
                decode_audio=False,
                split_type='train',
                split_id=self.data_split,
                transform=self.test_transform
            )
        self._calculate_index_to_prompt()
        self._calculate_index_to_label()
        print()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.hmdb51_train,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.hmdb51_val,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.hmdb51_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def _calculate_index_to_prompt(self):
        self.index_to_prompt = {
            idx: (self.prompt_prefix
                  + self.class_to_prompt[self.id_to_class[idx]].lower())
            for idx in range(len(self.id_to_class))
        }
        self.prompts = list(self.class_to_prompt.values())

    def _calculate_index_to_label(self):
        self.index_to_label = {
            idx: TemporalLabel.TEMPORAL
            for idx in range(len(self.id_to_class.keys()))
        }


if __name__ == '__main__':
    datamodule = HMDB51DataModule(
        data_root="/home/jochem/Documents/ai/scriptie/data",
        train_batch_size=2,
        test_batch_size=2,
        num_workers=4,
        nr_frames=4,
        prompt_prefix="",
        fps=4
    )
    datamodule.setup(stage='test')
    a = next(datamodule.hmdb51_test)
    print()
