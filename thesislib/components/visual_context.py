import pytorch_lightning as pl
import torch.nn
import torchvision
from . import S3D


class VisualContext(pl.LightningModule):
    def __init__(self, nr_output_vectors, vector_dim, video_resolution):
        super().__init__()
        self.nr_output_vectors = nr_output_vectors
        self.vector_dim = vector_dim
        self.video_resolution = video_resolution
        self.s3d = S3D(nr_output_vectors * vector_dim)

        self.pre_model_transforms = torch.nn.Sequential(
            torchvision.transforms.Resize(video_resolution)
        )

    def forward(self, frames):
        batch_size, nr_frames, *image_dims = frames.shape
        frames = self.pre_model_transforms(frames.view(-1, *image_dims))
        frames = frames.view(
            batch_size, nr_frames, *frames.shape[-3:]).permute(0, 2, 1, 3, 4)
        flat_embeddings = self.s3d(frames)
        return flat_embeddings.reshape(
            -1, self.nr_output_vectors, self.vector_dim)
