import torch
import pytorch_lightning as pl
from clip.simple_tokenizer import SimpleTokenizer


class TextualDomainAdaptation(pl.LightningModule):

    eot_token = SimpleTokenizer().encoder["<|endoftext|>"]

    def __init__(
            self,
            embedding_dim,
            da_length,
            da_insertion,
    ):
        super().__init__()
        da_vectors = torch.empty(da_length, embedding_dim, dtype=self.dtype)
        torch.nn.init.normal_(da_vectors, std=0.02)
        self.da_vectors = torch.nn.Parameter(da_vectors)
        self.da_length = da_length
        self.da_insertion = da_insertion

    def _insert_domain_adaptation(self, embeddings, eot_indices):

        if self.da_insertion == 'prefix':
            return self._insert_prefix_vectors(embeddings)
        elif self.da_insertion == 'postfix':
            return self._insert_postfix_vectors(embeddings,
                                                eot_indices)
        elif self.da_insertion == 'infix':
            assert self.da_length % 2 == 0
            return self._insert_infix_vectors(embeddings,
                                              eot_indices)

    def _insert_prefix_vectors(self, embeddings):
        batch_size = embeddings.shape[0]
        addition_repeated = self.da_vectors.repeat(batch_size, 1, 1)
        corrected_embeddings = torch.cat(
            [embeddings[:, :1, :],
             addition_repeated,
             embeddings[:, 1:(77 - len(self.da_vectors)), :]],
            dim=1
        )

        return corrected_embeddings

    def _insert_postfix_vectors(self, embeddings, eot_indices):
        batch_size = embeddings.shape[0]
        corrected_embeddings = torch.zeros_like(embeddings).type_as(embeddings)
        for idx in range(batch_size):
            eot_index = eot_indices[idx].item()
            corrected_embeddings[idx, :, :] = torch.cat(
                [embeddings[idx, :eot_index, :],
                 self.da_vectors,
                 embeddings[idx, eot_index:(77 - len(self.da_vectors)), :]]
            )

        return corrected_embeddings

    def _insert_infix_vectors(self, embeddings, eot_indices):
        batch_size = embeddings.shape[0]
        partial_length = len(self.da_vectors) // 2
        corrected_embeddings = torch.zeros_like(embeddings).type_as(embeddings)
        for idx in range(batch_size):
            eot_index = eot_indices[idx].item()
            corrected_embeddings[idx, :, :] = torch.cat(
                [embeddings[idx, :1, :],
                 self.da_vectors[:partial_length, :],
                 embeddings[idx, 1:eot_index, :],
                 self.da_vectors[partial_length:, :],
                 embeddings[idx, eot_index:(77 - len(self.da_vectors)), :]]
            )

        return corrected_embeddings

    def forward(self, text_embeddings, eot_indices):
        x = self._insert_domain_adaptation(text_embeddings,
                                           eot_indices=eot_indices)
        eot_indices += self.da_length
        return x, eot_indices
