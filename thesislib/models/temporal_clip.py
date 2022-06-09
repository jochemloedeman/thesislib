from typing import Dict, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer

from thesislib.models import ContextAddition, DomainAdaptation
from thesislib.temporal import TemporalLabel


class TemporalCLIP(pl.LightningModule):
    eot_token = SimpleTokenizer().encoder["<|endoftext|>"]

    def __init__(
            self,
            clip_architecture: str,
            da_settings: Union[Dict, None],
            ca_settings: Union[Dict, None],
            temporal_dataset: Dict

    ) -> None:
        super().__init__()

        self.clip_model, _ = clip.load(clip_architecture, device='cpu')
        self.temporal_dataset = temporal_dataset
        if not ca_settings:
            self.context_addition = None
        else:
            self.context_addition = ContextAddition(
                embedding_dim=self.clip_model.token_embedding.embedding_dim,
                **ca_settings
            )

        if not da_settings:
            self.domain_adaptation = None
        else:
            self.domain_adaptation = DomainAdaptation(
                embedding_dim=self.clip_model.token_embedding.embedding_dim,
                **da_settings
            )
        self.index_to_prompt = None
        self.index_to_label = None
        self._freeze_components()

        self.top1_accuracy = torchmetrics.Accuracy(
            num_classes=self.temporal_dataset['number_of_classes'],
            average='none',
        )
        self.top5_accuracy = torchmetrics.Accuracy(
            num_classes=self.temporal_dataset['number_of_classes'],
            top_k=5
        )

    def forward(self, frames):
        batch_size, nr_frames, channels, height, width = frames.size()
        flattened_frames = frames.reshape(-1, channels, height, width)

        video_features = self.encode_image(flattened_frames)
        video_features = video_features.reshape(batch_size, nr_frames, -1)
        video_features = video_features.mean(dim=1)
        video_features = video_features / video_features.norm(dim=1,
                                                              keepdim=True)

        text_features = self.encoded_text
        text_features = text_features / text_features.norm(dim=1,
                                                           keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_video = logit_scale * video_features @ text_features.t()
        logits_per_text = logits_per_video.t()
        return logits_per_video, logits_per_text

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-3
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=10
        )
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'validation_total_r@5',
                }}

    def training_step(self, batch):
        pass

    def validation_step(self, batch):
        pass

    def test_step(self, batch, batch_idx):
        frames, labels = batch
        logits_per_video, logits_per_text = self(frames)
        preds_best = logits_per_video.argmax(dim=-1)
        self.top5_accuracy(logits_per_video, labels)
        self.top1_accuracy(preds_best, labels)

    def test_epoch_end(self, outputs) -> None:
        acc_per_class = self.top1_accuracy.compute()
        total_acc = acc_per_class.mean()
        temporal_acc = acc_per_class[self.temporal_dataset['temporal']].mean()
        static_acc = acc_per_class[self.temporal_dataset['static']].mean()
        self.log('top1_accuracy_temporal', temporal_acc)
        self.log('top1_accuracy_static', static_acc)
        self.log('top1_accuracy_total', total_acc)
        self.log('top5_accuracy_total', self.top5_accuracy)

    def _encode_text(self) -> None:
        eot_indices = (self.tokenized_prompts
                       == self.eot_token).nonzero(as_tuple=True)[1]

        x = self.clip_model.token_embedding(self.tokenized_prompts)

        if self.domain_adaptation:
            x, eot_indices = self.domain_adaptation(x, eot_indices)

        if self.context_addition:
            dynamic_bools = [
                label == TemporalLabel.TEMPORAL
                for label in list(self.index_to_label.values())
            ]
            x, eot_indices = self.context_addition(
                x,
                eot_indices,
                dynamic_bools
            )

        x = x + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width] take features from
        # the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eot_indices]

        x = x @ self.clip_model.text_projection

        self.encoded_text = x

    def encode_image(self, image):
        return self.clip_model.visual(image.type(self.dtype))

    def _tokenize_classes(self) -> None:
        class_prompts = list(self.index_to_prompt.values())
        class_prompts = [
            "a video of someone who is " + prompt.lower()
            for prompt in class_prompts
        ]
        tokenized_prompts = clip.tokenize(class_prompts)
        self.tokenized_prompts = tokenized_prompts.to(self.device)

    def _freeze_components(self) -> None:
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def on_test_start(self) -> None:
        self.index_to_prompt = self.trainer.datamodule.index_to_prompt
        self.index_to_label = self.trainer.datamodule.index_to_label
        self._tokenize_classes()
        self._encode_text()

    def on_fit_start(self) -> None:
        self._tokenize_classes()
        self._encode_text()
