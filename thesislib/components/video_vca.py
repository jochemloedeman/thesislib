import pytorch_lightning as pl
import torch
import torchvision
from . import S3D


class VideoVCA(pl.LightningModule):
    def __init__(
            self,
            nr_output_vectors,
            vector_dim,
            video_resolution=112,
            input_type='diff',
            **kwargs,
    ) -> None:
        super().__init__()
        self.nr_output_vectors = nr_output_vectors
        self.vector_dim = vector_dim
        self.video_resolution = video_resolution
        self.input_type = input_type
        self.s3d = S3D(nr_output_vectors * vector_dim)

    def forward(self, frames):
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
