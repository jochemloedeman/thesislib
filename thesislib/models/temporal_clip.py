from typing import Dict, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
from torch.nn.functional import cross_entropy

from ..components import ContextAddition, DomainAdaptation, VisualContext
from ..temporal import TemporalLabel


class TemporalCLIP(pl.LightningModule):
    eot_token = SimpleTokenizer().encoder["<|endoftext|>"]

    def __init__(
            self,
            clip_architecture: str,
            nr_pred_frames: int,
            nr_context_frames: int,
            da_settings: Union[Dict, None],
            ca_settings: Union[Dict, None],
            vc_settings: Union[Dict, None],
            temporal_dataset: Dict

    ) -> None:
        super().__init__()

        self.clip_model, _ = clip.load(clip_architecture, device='cpu')
        self.temporal_dataset = temporal_dataset
        self.nr_pred_frames = nr_pred_frames
        self.nr_context_frames = nr_context_frames
        self._get_pred_frames()
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

        if not vc_settings:
            self.visual_context = None
        else:
            self.visual_context = VisualContext(
                **vc_settings
            )

        self.index_to_prompt = None
        self.index_to_label = None
        self._freeze_components()

        self.classwise_top1_accuracy = torchmetrics.Accuracy(
            num_classes=self.temporal_dataset['number_of_classes'],
            average='none',
            top_k=1
        )
        self.classwise_top5_accuracy = torchmetrics.Accuracy(
            num_classes=self.temporal_dataset['number_of_classes'],
            average='none',
            top_k=5
        )
        self.top5_accuracy = torchmetrics.Accuracy(
            top_k=5
        )
        self.top1_accuracy = torchmetrics.Accuracy(
            top_k=1
        )

    def forward(self, frames):
        video_features = self.encode_image(frames)
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
                    'monitor': 'val_loss',
                }}

    def training_step(self, batch):
        frames, labels = batch
        logits_per_video, logits_per_text = self(frames)
        image_loss = cross_entropy(logits_per_video, labels)
        text_loss = cross_entropy(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch):
        frames, labels = batch
        logits_per_video, logits_per_text = self(frames)
        image_loss = cross_entropy(logits_per_video, labels)
        text_loss = cross_entropy(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        self.log('val_loss', loss)
        self.top1_accuracy(logits_per_video, labels)
        self.top5_accuracy(logits_per_video, labels)
        self.classwise_top1_accuracy(logits_per_video, labels)
        self.classwise_top5_accuracy(logits_per_video, labels)

    def test_step(self, batch, batch_idx):
        frames, labels = batch
        logits_per_video, logits_per_text = self(frames)
        self.top1_accuracy(logits_per_video, labels)
        self.top5_accuracy(logits_per_video, labels)
        self.classwise_top1_accuracy(logits_per_video, labels)
        self.classwise_top5_accuracy(logits_per_video, labels)

    def test_epoch_end(self, outputs) -> None:
        top1_acc_per_class = self.classwise_top1_accuracy.compute()
        top5_acc_per_class = self.classwise_top5_accuracy.compute()

        temporal_top1_acc = top1_acc_per_class[
            self.temporal_dataset['temporal']].mean()
        static_top1_acc = top1_acc_per_class[
            self.temporal_dataset['static']].mean()

        temporal_top5_acc = top5_acc_per_class[
            self.temporal_dataset['temporal']].mean()
        static_top5_acc = top5_acc_per_class[
            self.temporal_dataset['static']].mean()

        self.log('test_top1_accuracy_temporal', temporal_top1_acc)
        self.log('test_top1_accuracy_static', static_top1_acc)

        self.log('test_top5_accuracy_temporal', temporal_top5_acc)
        self.log('test_top5_accuracy_static', static_top5_acc)

        self.log('test_top1_accuracy_total', self.top1_accuracy)
        self.log('test_top5_accuracy_total', self.top5_accuracy)

    def validation_epoch_end(self, outputs) -> None:
        acc_per_class = self.classwise_top1_accuracy.compute()
        temporal_acc = acc_per_class[self.temporal_dataset['temporal']].mean()
        static_acc = acc_per_class[self.temporal_dataset['static']].mean()
        self.log('val_accuracy_temporal', temporal_acc)
        self.log('val_accuracy_static', static_acc)
        self.log('val_top1_accuracy_total', self.top1_accuracy)
        self.log('val_top5_accuracy_total', self.top5_accuracy)

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

        self.encoded_text = self._modified_text_encode(x, eot_indices)

    def _modified_text_encode(self, x, eot_indices):
        x = x + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width] take features from
        # the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eot_indices]

        x = x @ self.clip_model.text_projection
        return x

    def encode_image(self, video):
        if video.dim == 4:
            video = video.unsqueeze(0)
        pred_frames = video[:, self.pred_frames]
        if self.visual_context:
            visual_context = self.visual_context(
                video.permute(0, 2, 1, 3, 4)
            )
            video_features = self._modified_visual_encode(pred_frames,
                                                          visual_context)
        else:
            video_features = self._modified_visual_encode(pred_frames)

        video_features = video_features.reshape(len(video),
                                                self.nr_pred_frames,
                                                -1)
        return video_features

    def _modified_visual_encode(self, x, context=None):
        batch_size, nr_frames, channels, height, width = x.size()
        x = self.clip_model.visual.conv1(x.view(-1, channels, height, width))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.clip_model.visual.class_embedding.to(x.dtype)
             + torch.zeros(x.shape[0], 1, x.shape[-1],
                           dtype=x.dtype, device=x.device),
             x],
            dim=1
        )
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        if context is not None:
            context = (context.unsqueeze(1)
                       + torch.zeros(batch_size, nr_frames,
                                     self.visual_context.nr_output_vectors,
                                     x.shape[-1]).type_as(x)
                       )

            x = torch.cat(
                [x.view(batch_size, nr_frames, *x.shape[1:]), context],
                dim=2
            )
            x = x.reshape(-1, *x.shape[2:])
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip_model.visual.ln_post(x[:, 0, :])

        if self.clip_model.visual.proj is not None:
            x = x @ self.clip_model.visual.proj

        return x

    def _get_pred_frames(self):
        if self.nr_pred_frames == 1:
            pred_frames = [self.nr_context_frames // 2]
        elif self.nr_pred_frames == 2:
            pred_frames = [self.nr_context_frames // 3,
                           self.nr_context_frames // (3 / 2)]
        else:
            pred_frames = np.linspace(start=0,
                                      stop=self.nr_context_frames - 1,
                                      num=self.nr_pred_frames,
                                      endpoint=True).tolist()

        self.pred_frames = pred_frames

    def _tokenize_classes(self) -> None:
        class_prompts = list(self.index_to_prompt.values())
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
