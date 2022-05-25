import pytorch_lightning
import torch
import pytorch_lightning as pl
from clip.simple_tokenizer import SimpleTokenizer
from torch import nn


class ContextAddition(pl.LightningModule):
    def __init__(
            self,
            clip_model,
            ca_length,
            ca_insertion,
            target_partition
    ):

        super().__init__()
        embedding_dim = clip_model.token_embedding.embedding_dim
        ca_vectors = torch.empty(ca_length, embedding_dim, dtype=self.dtype)
        nn.init.normal_(ca_vectors, std=0.02)
        self.ca_vectors = nn.Parameter(ca_vectors)
        self.ca_length = ca_length
        self.ca_insertion = ca_insertion
        self.target_partition = target_partition
        self.eot_token = SimpleTokenizer().encoder["<|endoftext|>"]

    def _insert_context(self, embeddings, dynamic_bools, eot_indices):

        if self.target_partition == 'all':
            dynamic_bools = torch.tensor([True] * len(dynamic_bools)).type_as(
                dynamic_bools)
        elif self.target_partition == 'none':
            dynamic_bools = torch.tensor([False] * len(dynamic_bools)).type_as(
                dynamic_bools)

        if self.ca_insertion == 'prefix':
            return self._insert_prefix_vectors(embeddings, dynamic_bools)
        elif self.ca_insertion == 'postfix':
            return self._insert_postfix_vectors(embeddings, dynamic_bools,
                                                eot_indices)
        elif self.ca_insertion == 'infix':
            assert self.ca_length % 2 == 0
            return self._insert_infix_vectors(embeddings, dynamic_bools,
                                              eot_indices)

    def _insert_prefix_vectors(self, embeddings, sample_selection):
        batch_size = embeddings.shape[0]
        addition_repeated = self.ca_vectors.repeat(batch_size, 1, 1)
        all_dynamic = torch.cat(
            [embeddings[:, :1, :],
             addition_repeated,
             embeddings[:, 1:(77 - len(self.ca_vectors)), :]],
            dim=1)

        sample_selection = sample_selection.reshape(-1, 1, 1)
        corrected_embeddings = torch.where(sample_selection, all_dynamic,
                                           embeddings)
        return corrected_embeddings

    def _insert_postfix_vectors(self, embeddings, sample_selection, eot_indices):
        batch_size = embeddings.shape[0]
        all_dynamic = torch.zeros_like(embeddings).type_as(embeddings)
        for idx in range(batch_size):
            eot_index = eot_indices[idx].item()
            all_dynamic[idx, :, :] = torch.cat(
                [embeddings[idx, :eot_index, :],
                 self.ca_vectors,
                 embeddings[idx, eot_index:(77 - len(self.ca_vectors)), :]]
            )
        sample_selection = sample_selection.reshape(-1, 1, 1)
        corrected_embeddings = torch.where(sample_selection, all_dynamic,
                                           embeddings)
        return corrected_embeddings

    def _insert_infix_vectors(self, embeddings, sample_selection, eot_indices):
        batch_size = embeddings.shape[0]
        partial_length = len(self.ca_vectors) // 2
        all_dynamic = torch.zeros_like(embeddings).type_as(embeddings)
        for idx in range(batch_size):
            eot_index = eot_indices[idx].item()
            all_dynamic[idx, :, :] = torch.cat(
                [embeddings[idx, :1, :],
                 self.ca_vectors[:partial_length, :],
                 embeddings[idx, 1:eot_index, :],
                 self.ca_vectors[partial_length:, :],
                 embeddings[idx, eot_index:(77 - len(self.ca_vectors)), :]]
            )

        sample_selection = sample_selection.reshape(-1, 1, 1)
        corrected_embeddings = torch.where(sample_selection, all_dynamic,
                                           embeddings)
        return corrected_embeddings

    def forward(self, text_embeddings, eot_indices, dynamic_bools):
        x = self._insert_context(text_embeddings, dynamic_bools, eot_indices)
        return x
