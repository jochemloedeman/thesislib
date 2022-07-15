import pathlib
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import pytorchvideo.transforms
import torch
import torchvision.transforms
from pytorchvideo.data import Kinetics
from torch.utils.data import DataLoader

from thesislib.datasets import MSRVTT

class MSRVTTDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root,
            train_batch_size,
            test_batch_size,
            num_workers,
            nr_frames,
            prompt_prefix,
            fps,
            **kwargs,
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
        root_dir = pathlib.Path(self.data_root) / 'msrvtt'
        index_to_caption_path = (root_dir /
                                 'annotations' /
                                 'MSRVTT_JSFUSION_test_debug.csv')
        annotations = pd.read_csv(index_to_caption_path).to_dict()
        self.index_to_caption = annotations['sentence']
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
                torchvision.transforms.RandomCrop(size=224),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
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
                torchvision.transforms.CenterCrop(size=224),
            ])
        )
        if stage == 'fit':
            self.msrvtt_train = MSRVTT(
                data_path=(
                        root_dir / 'annotations' / 'train.csv').as_posix(),
                clip_sampler=pytorchvideo.data.RandomClipSampler(
                    clip_duration=float(self.nr_frames / self.fps)
                ),
                video_sampler=torch.utils.data.DistributedSampler,
                video_path_prefix=(root_dir / 'train').as_posix(),
                decode_audio=False,
                transform=self.train_transform
            )
            self.msrvtt_val = MSRVTT(
                data_path=(
                        root_dir / 'annotations' / 'validate.csv')
                .as_posix(),
                clip_sampler=pytorchvideo.data.RandomClipSampler(
                    clip_duration=float(self.nr_frames / self.fps)
                ),
                video_sampler=torch.utils.data.DistributedSampler,
                video_path_prefix=(root_dir / 'val').as_posix(),
                decode_audio=False,
                transform=self.test_transform
            )
        if stage == 'test':
            self.msrvtt_test = MSRVTT(
                data_path=(root_dir / 'annotations' / 'test_debug.csv').as_posix(),
                clip_sampler=pytorchvideo.data.UniformClipSampler(
                    clip_duration=float(self.nr_frames / self.fps)
                ),
                video_sampler=torch.utils.data.SequentialSampler,
                video_path_prefix=(root_dir / 'test_videos').as_posix(),
                decode_audio=False,
                transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.msrvtt_train,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.msrvtt_val,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.msrvtt_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

if __name__ == '__main__':
    datamodule = MSRVTTDataModule(
        data_root="/home/jochem/Documents/ai/scriptie/data",
        train_batch_size=2,
        test_batch_size=2,
        num_workers=0,
        nr_frames=8,
        prompt_prefix="",
        fps=8,
    )
    datamodule.setup(stage="test")
    a = datamodule.msrvtt_test.__next__()
    print()
