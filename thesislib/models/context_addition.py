import torch
from clip.simple_tokenizer import SimpleTokenizer
from torch import nn


class ContextAddition(nn.Module):
    def __init__(self,
                 clip_model,
                 ca_length,
                 ca_insertion,
                 da_length,
                 da_insertion,
                 target_partition):

        super().__init__()
        dtype = clip_model.dtype
        embedding_dim = clip_model.token_embedding.embedding_dim
        ca_vectors = torch.empty(ca_length, embedding_dim, dtype=dtype)
        da_vectors = torch.empty(da_length, embedding_dim, dtype=dtype)
        nn.init.normal_(ca_vectors, std=0.02)
        self.ca_vectors = nn.Parameter(ca_vectors)
        self.da_vectors = nn.Parameter(da_vectors)
        self.ca_length = ca_length
        self.ca_insertion = ca_insertion
        self.da_length = da_length
        self.da_insertion = da_insertion
        self.target_partition = target_partition
        self.token_embedding = clip_model.token_embedding
        self.eot_token = SimpleTokenizer().encoder["<|endoftext|>"]

    def _insert_context(self, embeddings, dynamic_bools, tokenized_text):

        if self.target_partition == 'all':
            dynamic_bools = torch.tensor([True] * len(dynamic_bools)).type_as(dynamic_bools)
        elif self.target_partition == 'none':
            dynamic_bools = torch.tensor([False] * len(dynamic_bools)).type_as(dynamic_bools)

        if self.ca_insertion == 'prefix':
            return self._insert_prefix_context(embeddings, dynamic_bools)
        elif self.ca_insertion == 'postfix':
            return self._insert_postfix_context(embeddings, dynamic_bools, tokenized_text)
        elif self.ca_insertion == 'infix':
            assert self.ca_length % 2 == 0
            return self._insert_infix_context(embeddings, dynamic_bools, tokenized_text)

    # def _insert_domain_adaptation(self, embeddings, tokenized_text):
    #     if self.da_insertion == 'prefix':
    #     elif self.da_insertion == 'postfix':
    #     elif self.da_insertion == 'infix':
    #         assert self.da_length % 2 == 0

    def _insert_prefix_context(self, embeddings, dynamic_bools):
        batch_size = embeddings.shape[0]
        context_block = self.ca_vectors.repeat(batch_size, 1, 1)
        all_dynamic = torch.cat([embeddings[:, :1, :],
                                 context_block,
                                 embeddings[:, 1:(77 - self.ca_length), :]], dim=1)

        dynamic_bools = dynamic_bools.reshape(-1, 1, 1)
        corrected_embeddings = torch.where(dynamic_bools, all_dynamic, embeddings)
        return corrected_embeddings

    def _insert_postfix_context(self, embeddings, dynamic_bools, tokenized_text):
        batch_size = embeddings.shape[0]
        all_dynamic = torch.zeros_like(embeddings).type_as(embeddings)
        context_block = self.ca_vectors
        eot_indices = (tokenized_text == self.eot_token).nonzero(as_tuple=True)[1]
        for idx in range(batch_size):
            eot_index = eot_indices[idx].item()
            all_dynamic[idx, :, :] = torch.cat([embeddings[idx, :eot_index, :],
                                                context_block,
                                                embeddings[idx, eot_index:(77 - self.ca_length), :]
                                                ])
        dynamic_bools = dynamic_bools.reshape(-1, 1, 1)
        corrected_embeddings = torch.where(dynamic_bools, all_dynamic, embeddings)
        return corrected_embeddings

    def _insert_infix_context(self, embeddings, dynamic_bools, tokenized_text):
        batch_size = embeddings.shape[0]
        partial_length = self.ca_length // 2
        all_dynamic = torch.zeros_like(embeddings).type_as(embeddings)
        context_block = self.ca_vectors
        eot_indices = (tokenized_text == self.eot_token).nonzero(as_tuple=True)[1]
        for idx in range(batch_size):
            eot_index = eot_indices[idx].item()
            all_dynamic[idx, :, :] = torch.cat([embeddings[idx, :1, :],
                                                context_block[:partial_length, :],
                                                embeddings[idx, 1:eot_index, :],
                                                context_block[partial_length:, :],
                                                embeddings[idx, eot_index:(77 - self.ca_length), :]
                                                ])

        dynamic_bools = dynamic_bools.reshape(-1, 1, 1)
        corrected_embeddings = torch.where(dynamic_bools, all_dynamic, embeddings)
        return corrected_embeddings

    # def _insert_prefix_vectors(self, vectors, bools):

    def forward(self, tokenized_text, dynamic_bools):
        x = self.token_embedding(tokenized_text)  # [batch_size, n_ctx, d_model]
        x = self._insert_context(x, dynamic_bools, tokenized_text)
        return x
