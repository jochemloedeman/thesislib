import random

import pytorch_lightning as pl
import torch
import torchvision


class ImageVCA(pl.LightningModule):
    def __init__(
            self,
            nr_output_vectors,
            vector_dim,
            pretrained,
            video_resolution=112,
            image_sample_mode='center',
            **kwargs,
    ) -> None:
        super().__init__()
        self.nr_output_vectors = nr_output_vectors
        self.vector_dim = vector_dim
        self.video_resolution = video_resolution
        self.sample_mode = image_sample_mode
        self.model = self._generate_resnet(pretrained)

    def forward(self, frames):
        frames = self._resize(frames)
        input_frame = self._sample_frame(frames)
        flat_embeddings = self.model(input_frame)
        return flat_embeddings.reshape(
            -1, self.nr_output_vectors, self.vector_dim)

    def _sample_frame(self, frames):
        if self.sample_mode == 'random':
            frame_idx = random.randint(0, frames.shape[2] - 1)
            return frames[:, :, frame_idx, :, :].squeeze()
        else:
            return frames[:, :, frames.shape[2] // 2, :, :].squeeze()

    def _resize(self, frames):
        frame_shape = frames.shape
        frames = torchvision.transforms.functional.resize(
            frames.view(-1, *frame_shape[-2:]),
            size=(self.video_resolution, self.video_resolution)
        )
        return frames.view(*frame_shape[:3],
                           self.video_resolution,
                           self.video_resolution)

    def _generate_resnet(self, pretrained):
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        resnet.fc = torch.nn.Linear(
            in_features=512,
            out_features=self.nr_output_vectors * self.vector_dim
        )
        return resnet
