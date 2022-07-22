import pathlib
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import pytorchvideo.transforms
import torch
import torchvision.transforms
from pytorchvideo.data import Kinetics
from torch.utils.data import DataLoader
from thesislib.temporal import TemporalLabel


class Kinetics400DataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root,
            train_batch_size,
            test_batch_size,
            num_workers,
            nr_frames,
            fps,
            temporal_dataset,
            **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.nr_frames = nr_frames
        self.fps = fps
        self.temporal_dataset = temporal_dataset

        self.prompt_prefixes = [
            "A video of", 
            "A video of someone", 
            "A video of a person", 
            "A photo of", 
            "A photo of someone", 
            "A photo of a person"
        ]

    def setup(self, stage: Optional[str] = None) -> None:
        root_dir = pathlib.Path(self.data_root) / 'kinetics'
        labels_to_id = root_dir / 'annotations' / 'labels_to_id.csv'
        self.id_to_class = pd.read_csv(labels_to_id).to_dict()['name']
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}
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
            self.kinetics_train = Kinetics(
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
            self.kinetics_val = Kinetics(
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
            self.kinetics_test = Kinetics(
                data_path=(root_dir / 'annotations' / 'test.csv').as_posix(),
                clip_sampler=pytorchvideo.data.UniformClipSampler(
                    clip_duration=float(self.nr_frames / self.fps)
                ),
                video_sampler=torch.utils.data.SequentialSampler,
                video_path_prefix=(root_dir / 'test').as_posix(),
                decode_audio=False,
                transform=self.test_transform
            )

        self._calculate_index_to_classes()
        print()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.kinetics_train,
            batch_size=self.train_batch_size,
            shuffle=False,
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

    def _calculate_index_to_classes(self):
        classes = [
            class_str.replace("_", " ")
            for class_str in self.id_to_class.values()
        ]
        self.index_to_classes = {
            idx: classes[idx] for idx in range(len(classes))
        }


if __name__ == '__main__':
    module = Kinetics400DataModule(
        test_batch_size=1,
        num_workers=4
    )
    module.setup()
    print()
