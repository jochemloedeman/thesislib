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
        nn.init.normal_(da_vectors, std=0.02)
        self.ca_vectors = nn.Parameter(ca_vectors)
        self.da_vectors = nn.Parameter(da_vectors)
        self.ca_length = ca_length
        self.ca_insertion = ca_insertion
        self.da_length = da_length
        self.da_insertion = da_insertion
        self.target_partition = target_partition
        self.token_embedding = clip_model.token_embedding
        self.eot_token = SimpleTokenizer().encoder["<|endoftext|>"]

    def _insert_context(self, embeddings, dynamic_bools, eot_indices):

        if self.target_partition == 'all':
            dynamic_bools = torch.tensor([True] * len(dynamic_bools)).type_as(dynamic_bools)
        elif self.target_partition == 'none':
            dynamic_bools = torch.tensor([False] * len(dynamic_bools)).type_as(dynamic_bools)

        if self.ca_insertion == 'prefix':
            return self._insert_prefix_vectors(embeddings, dynamic_bools, self.ca_vectors)
        elif self.ca_insertion == 'postfix':
            return self._insert_postfix_vectors(embeddings, dynamic_bools, eot_indices, self.ca_vectors)
        elif self.ca_insertion == 'infix':
            assert self.ca_length % 2 == 0
            return self._insert_infix_vectors(embeddings, dynamic_bools, eot_indices, self.ca_vectors)

    def _insert_domain_adaptation(self, embeddings, eot_indices):

        sample_selection = torch.tensor([True] * len(embeddings), device=embeddings.device)

        if self.ca_insertion == 'prefix':
            return self._insert_prefix_vectors(embeddings, sample_selection, self.da_vectors)
        elif self.ca_insertion == 'postfix':
            return self._insert_postfix_vectors(embeddings, sample_selection, eot_indices, self.da_vectors)
        elif self.ca_insertion == 'infix':
            assert self.da_length % 2 == 0
            return self._insert_infix_vectors(embeddings, sample_selection, eot_indices, self.da_vectors)

    def _insert_prefix_vectors(self, embeddings, sample_selection, addition_vectors):
        batch_size = embeddings.shape[0]
        addition_repeated = addition_vectors.repeat(batch_size, 1, 1)
        all_dynamic = torch.cat([embeddings[:, :1, :],
                                 addition_repeated,
                                 embeddings[:, 1:(77 - len(addition_vectors)), :]], dim=1)

        sample_selection = sample_selection.reshape(-1, 1, 1)
        corrected_embeddings = torch.where(sample_selection, all_dynamic, embeddings)
        return corrected_embeddings

    def _insert_postfix_vectors(self, embeddings, sample_selection, eot_indices, addition_vectors):
        batch_size = embeddings.shape[0]
        all_dynamic = torch.zeros_like(embeddings).type_as(embeddings)
        for idx in range(batch_size):
            eot_index = eot_indices[idx].item()
            all_dynamic[idx, :, :] = torch.cat([embeddings[idx, :eot_index, :],
                                                addition_vectors,
                                                embeddings[idx, eot_index:(77 - len(addition_vectors)), :]
                                                ])
        sample_selection = sample_selection.reshape(-1, 1, 1)
        corrected_embeddings = torch.where(sample_selection, all_dynamic, embeddings)
        return corrected_embeddings

    def _insert_infix_vectors(self, embeddings, sample_selection, eot_indices, addition_vectors):
        batch_size = embeddings.shape[0]
        partial_length = len(addition_vectors) // 2
        all_dynamic = torch.zeros_like(embeddings).type_as(embeddings)
        for idx in range(batch_size):
            eot_index = eot_indices[idx].item()
            all_dynamic[idx, :, :] = torch.cat([embeddings[idx, :1, :],
                                                addition_vectors[:partial_length, :],
                                                embeddings[idx, 1:eot_index, :],
                                                addition_vectors[partial_length:, :],
                                                embeddings[idx, eot_index:(77 - len(addition_vectors)), :]
                                                ])

        sample_selection = sample_selection.reshape(-1, 1, 1)
        corrected_embeddings = torch.where(sample_selection, all_dynamic, embeddings)
        return corrected_embeddings

    # def _insert_prefix_vectors(self, vectors, bools):

    def forward(self, tokenized_text, dynamic_bools):
        eot_indices = (tokenized_text == self.eot_token).nonzero(as_tuple=True)[1]
        x = self.token_embedding(tokenized_text)  # [batch_size, n_ctx, d_model]
        x = self._insert_domain_adaptation(x, eot_indices=eot_indices)
        eot_indices += self.da_length
        x = self._insert_context(x, dynamic_bools, eot_indices)
        return x
