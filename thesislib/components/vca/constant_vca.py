import pytorch_lightning as pl
import torch


class ConstantVCA(pl.LightningModule):
    def __init__(
            self,
            nr_output_vectors,
            vector_dim,
            **kwargs,
    ) -> None:
        super().__init__()
        self.nr_output_vectors = nr_output_vectors
        self.vector_dim = vector_dim

        vca_vectors = torch.empty(
            nr_output_vectors,
            vector_dim,
            dtype=self.dtype,
            device=self.device,
        )
        torch.nn.init.normal_(vca_vectors, std=0.02)
        self.vca_vectors = torch.nn.Parameter(vca_vectors)

    def forward(self, frames):
        batch_size = len(frames)
        return self.vca_vectors.repeat(batch_size, 1, 1).type_as(frames)

    def set_train_transforms(self):
        pass

    def set_val_test_transforms(self):
        pass
