import pytorch_lightning as pl
import torch
import transformers

from thesislib.components import insert_tca_vectors


class LMTCA(pl.LightningModule):
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
        self._configure_lm()

    def _configure_lm(self):
        self.tokenizer = transformers.AlbertTokenizer.from_pretrained(
            "albert-base-v2"
        )
        self.lm = transformers.AlbertModel.from_pretrained("albert-base-v2")
        self.projection = torch.nn.Linear(
            in_features=self.lm.config.hidden_size,
            out_features=(self.nr_output_vectors * self.vector_dim)
        )

    def forward(self, text_embeddings, eot_indices, prompts):
        lm_inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True
        ).to(self.device)

        cls_embedding = self.lm(**lm_inputs).last_hidden_state[:, 0]
        context_tokens = self.projection(cls_embedding).reshape(
            -1, self.nr_output_vectors, self.vector_dim)

        extended_embeddings = insert_tca_vectors(
            context_vectors=context_tokens,
            embeddings=text_embeddings,
            eot_indices=eot_indices,
            mode=self.insertion_mode
        )
        eot_indices += self.nr_output_vectors
        return extended_embeddings.type_as(text_embeddings), eot_indices
