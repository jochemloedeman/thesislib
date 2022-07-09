import pytorch_lightning as pl
import torch

from . import insert_tca_vectors


class ConstantTCA(pl.LightningModule):
    def __init__(
            self,
            nr_output_vectors,
            vector_dim,
            insertion_mode,
            **kwargs,
    ) -> None:
        super().__init__()
        self.nr_output_vectors = nr_output_vectors
        self.vector_dim = vector_dim
        self.insertion_mode = insertion_mode

        tca_vectors = torch.empty(
            nr_output_vectors,
            vector_dim,
            dtype=self.dtype,
            device=self.device,
        )
        torch.nn.init.normal_(tca_vectors, std=0.02)
        self.tca_vectors = torch.nn.Parameter(tca_vectors)

    def forward(self, text_embeddings, eot_indices, prompts=None):
        batch_size = len(text_embeddings)
        vectors_repeated = self.tca_vectors.repeat(batch_size, 1, 1)
        extended_embeddings = insert_tca_vectors(
            context_vectors=vectors_repeated,
            embeddings=text_embeddings,
            eot_indices=eot_indices,
            mode=self.insertion_mode
        )
        eot_indices += self.nr_output_vectors
        return extended_embeddings.type_as(text_embeddings), eot_indices