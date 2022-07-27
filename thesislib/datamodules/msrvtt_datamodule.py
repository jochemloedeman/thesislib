from collections import defaultdict
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
            fps,
            **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.nr_frames = nr_frames
        self.fps = fps

    def setup(self, stage: Optional[str] = None) -> None:
        root_dir = pathlib.Path(self.data_root) / 'msrvtt'
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
            trainval_captions_path = (root_dir /
                                      'annotations' /
                                      'trainval_captions.csv')
            annotations = pd.read_csv(trainval_captions_path)
            self.index_to_caption = self.create_caption_index(annotations)
            self.txt2vis, self.vis2txt = self.create_index_mappings(annotations)
            self.msrvtt_train = MSRVTT(
                data_path=(
                        root_dir / 'annotations' / 'train.csv').as_posix(),
                clip_sampler=pytorchvideo.data.RandomClipSampler(
                    clip_duration=float(self.nr_frames / self.fps)
                ),
                video_sampler=torch.utils.data.DistributedSampler,
                video_path_prefix=(root_dir / 'videos').as_posix(),
                decode_audio=False,
                transform=self.train_transform
            )
            self.msrvtt_val = MSRVTT(
                data_path=(
                        root_dir / 'annotations' / 'val.csv')
                .as_posix(),
                clip_sampler=pytorchvideo.data.RandomClipSampler(
                    clip_duration=float(self.nr_frames / self.fps)
                ),
                video_sampler=torch.utils.data.DistributedSampler,
                video_path_prefix=(root_dir / 'videos').as_posix(),
                decode_audio=False,
                transform=self.test_transform
            )
        if stage == 'test':
            test_captions_path = (root_dir /
                                  'annotations' /
                                  'test_captions.csv')

            annotations = pd.read_csv(test_captions_path)
            self.index_to_caption = self.create_caption_index(annotations)
            self.txt2vis, self.vis2txt = self.create_index_mappings(annotations)

            self.msrvtt_test = MSRVTT(
                data_path=(root_dir / 'annotations' / 'test.csv').as_posix(),
                clip_sampler=pytorchvideo.data.UniformClipSampler(
                    clip_duration=float(self.nr_frames / self.fps)
                ),
                video_sampler=torch.utils.data.SequentialSampler,
                video_path_prefix=(root_dir / 'videos').as_posix(),
                decode_audio=False,
                transform=self.test_transform
            )

    @staticmethod
    def create_index_mappings(annotations: pd.DataFrame):
        vis2txt = defaultdict(list)
        txt2vis = {}
        for index, row in annotations.iterrows():
            vis2txt[row['video_id']] += [index]
        for key, values in vis2txt.items():
            for value in values:
                txt2vis[value] = key
        return txt2vis, vis2txt

    @staticmethod
    def create_caption_index(annotations: pd.DataFrame):
        captions = defaultdict(list)
        for index, row in annotations.iterrows():
            captions[row["video_id"]] += [row["sentence"]]

        return captions

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
