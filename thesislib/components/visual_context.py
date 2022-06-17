import pytorch_lightning as pl

from . import S3D


class VisualContext(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.s3d = S3D(num_classes)

    def forward(self, frames):
        return self.s3d(frames)