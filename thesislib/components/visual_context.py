import pytorch_lightning as pl

from . import S3D


class VisualContext(pl.LightningModule):
    def __init__(self, nr_output_vectors, vector_dim):
        super().__init__()
        self.nr_output_vectors = nr_output_vectors
        self.vector_dim = vector_dim
        self.s3d = S3D(nr_output_vectors * vector_dim)

    def forward(self, frames):
        flat_embeddings = self.s3d(frames)
        return flat_embeddings.reshape(-1, self.nr_output_vectors, self.vector_dim)