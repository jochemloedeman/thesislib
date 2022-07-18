import json
import pathlib
from typing import Optional

import pytorch_lightning as pl
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from thesislib.temporal import TemporalLabel


class SSV2DataModule(pl.LightningDataModule):
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
        root_dir = pathlib.Path(self.data_root) / 'something-something-v2'
        annotation_train_path = root_dir / 'something-something-v2-train-processed.txt'
        annotation_val_path = root_dir / 'something-something-v2-validation-processed.txt'
        labels_path = root_dir / 'labels.json'

        with open(labels_path, 'r') as label_file:
            self.labels = json.load(label_file)

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
        self.ssv2_test = VideoFrameDataset(
            root_path=root_dir.as_posix(),
            annotationfile_path=annotation_val_path.as_posix(),
            imagefile_template='{:06d}.jpg',
            num_segments=self.nr_frames,
            frames_per_segment=1,
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
            dataset=self.ssv2_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def _calculate_index_to_prompt(self):
        self.index_to_prompt = {
            int(index): self.prompt_prefix + prompt.lower()
            for prompt, index in self.labels.items()
        }

    def _calculate_index_to_label(self):
        self.index_to_label = {
            int(idx): TemporalLabel.TEMPORAL
            for idx in self.labels.values()
        }


if __name__ == '__main__':
    module = SSV2DataModule(
        test_batch_size=2,
        num_workers=4,
        frames_per_vid=4
    )
    module.setup()
    print()
