import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision

from . import S3D, S3DHD
from ...permutation import TemporalPermutation


class VideoVCA(pl.LightningModule):
    def __init__(
            self,
            nr_output_vectors,
            vector_dim,
            video_resolution=112,
            input_type='diff',
            temporal_permutation=False,
            **kwargs,
    ) -> None:
        super().__init__()
        self.nr_output_vectors = nr_output_vectors
        self.vector_dim = vector_dim
        self.video_resolution = video_resolution
        self.input_type = input_type
        if video_resolution == 224:
            s3d = self._load_model_from_pt()
        else:
            s3d = S3D(nr_output_vectors * vector_dim)

        self.s3d = s3d
        if temporal_permutation:
            self.temporal_permutation = TemporalPermutation()
        else:
            self.temporal_permutation = None

    def _load_model_from_pt(self):
        model = S3DHD(400)
        file_weight = Path(__file__).parent / 'S3D_kinetics400.pt'
        if os.path.isfile(file_weight):
            print('loading weight file')
            weight_dict = torch.load(file_weight)
            model_dict = model.state_dict()
            for name, param in weight_dict.items():
                if 'module' in name:
                    name = '.'.join(name.split('.')[1:])
                if name in model_dict:
                    if param.size() == model_dict[name].size():
                        model_dict[name].copy_(param)
                    else:
                        print(' size? ' + name, param.size(),
                              model_dict[name].size())
                else:
                    print(' name? ' + name)

            print(' loaded')
        else:
            print('weight file?')

        model.fc = torch.nn.Conv3d(
            1024, (self.nr_output_vectors * self.vector_dim),
            kernel_size=1, stride=1, bias=True
        )
        return model

    def forward(self, frames):
        if self.temporal_permutation:
            frames = self.temporal_permutation(frames)

        frames = self._resize(frames)

        if self.input_type == 'diff':
            frames = self._calculate_rgb_diff(frames)

        flat_embeddings = self.s3d(frames)
        return flat_embeddings.reshape(
            -1, self.nr_output_vectors, self.vector_dim)

    def _resize(self, frames):
        frame_shape = frames.shape
        frames = torchvision.transforms.functional.resize(
            frames.view(-1, *frame_shape[-2:]),
            size=(self.video_resolution, self.video_resolution)
        )
        return frames.view(*frame_shape[:3],
                           self.video_resolution,
                           self.video_resolution)

    def _calculate_rgb_diff(self, frames):
        left_frames = frames[:, :, 1:, :, :]
        right_frames = frames[:, :, :-1, :, :]
        diff = left_frames - right_frames
        return self._normalize(diff)

    @staticmethod
    def _normalize(frames):
        means = frames.mean(dim=(0, 2, 3, 4))
        stds = frames.std(dim=(0, 2, 3, 4))
        if torch.any(stds == 0):
            return frames
        normalized_frames = torchvision.transforms.functional.normalize(
            frames.permute(0, 2, 1, 3, 4),
            mean=means,
            std=stds
        )
        return normalized_frames.permute(0, 2, 1, 3, 4)
