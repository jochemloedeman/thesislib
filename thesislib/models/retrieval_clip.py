from typing import Dict, Union, Optional, List

import numpy as np
import pytorch_lightning as pl
import torch
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
from torch.nn.functional import cross_entropy

from ..components.tca import ConstantTCA, LMTCA
from ..components.vca import ConstantVCA, VideoVCA, ImageVCA
from ..metrics.retrieval_recall import RetrievalRecall
from ..permutation import VCAPermutation


class RetrievalCLIP(pl.LightningModule):
    eot_token = SimpleTokenizer().encoder["<|endoftext|>"]

    def __init__(
            self,
            clip_architecture: str,
            nr_pred_frames: int,
            nr_context_frames: int,
            vca_settings: Union[Dict, None],
            tca_settings: Union[Dict, None],
            optimizer: str,
            lr_scheduler: Optional[str] = 'cosine',
            epochs: Optional[int] = 150,
            permutation_mode: Optional[str] = None,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.clip_model, _ = clip.load(clip_architecture, device='cpu')
        self.nr_pred_frames = nr_pred_frames
        self.nr_context_frames = nr_context_frames
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self._get_pred_frames()
        self._build_permutation(permutation_mode)

        self.index_to_prompt = None
        self.index_to_label = None
        self.class_to_id = None
        self._freeze_components()
        self._build_vca_module(vca_settings)
        self._build_tca_module(tca_settings)
        self._create_validation_metrics()

    def _build_permutation(self, permutation_mode):
        if permutation_mode:
            self.vca_permutation = VCAPermutation(permutation_mode)
        else:
            self.vca_permutation = None

    def _build_tca_module(self, tca_settings):
        if not tca_settings:
            self.textual_context_addition = None
        elif tca_settings['tca_mode'] == 'lm':
            self.textual_context_addition = LMTCA(
                **tca_settings
            )
        else:
            self.textual_context_addition = ConstantTCA(
                **tca_settings
            )

    def _build_vca_module(self, vca_settings):
        if not vca_settings:
            self.visual_context_addition = None
        elif vca_settings['vca_mode'] == "video":
            self.visual_context_addition = VideoVCA(
                **vca_settings
            )
        elif vca_settings['vca_mode'] == 'image':
            self.visual_context_addition = ImageVCA(
                **vca_settings
            )
        else:
            self.visual_context_addition = ConstantVCA(
                **vca_settings
            )

    def _create_validation_metrics(self):
        pass

    def _create_test_metrics(self):
        self.retrieval_recall = RetrievalRecall(self.trainer.datamodule.txt2vis)

    def forward(self, frames: torch.Tensor, captions: List[str]):
        video_features = self._encode_image(frames)
        video_features = video_features.mean(dim=1)
        video_features = video_features / video_features.norm(dim=1,
                                                              keepdim=True)
        text_features = self._encode_text(captions)
        text_features = text_features / text_features.norm(dim=1,
                                                           keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_video = logit_scale * video_features @ text_features.t()
        logits_per_text = logits_per_video.t()
        return logits_per_video, logits_per_text

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=1e-3
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=1e-1,
                weight_decay=0.0001,
                momentum=0.9
            )
        if self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.epochs,
                verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=[80, 120]
            )
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                }}

    def training_step(self, batch, batch_idx):
        frames, labels, video_indices = (
            batch["video"],
            batch["label"],
            batch["video_index"].type(torch.int32)
        )
        labels = labels.tolist()
        captions = [self.index_to_caption[label] for label in labels]
        logits_per_video, logits_per_text = self(frames, captions)
        loss = cross_entropy(logits_per_video, labels)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        frames, labels = (
            batch["video"],
            batch["label"]
        )
        logits_per_video, logits_per_text = self(frames)
        loss = cross_entropy(logits_per_video, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.top1_accuracy(logits_per_video, labels)
        self.top5_accuracy(logits_per_video, labels)
        self.classwise_top1_accuracy(logits_per_video, labels)
        self.classwise_top5_accuracy(logits_per_video, labels)

    def test_step(self, batch, batch_idx):
        frames, labels, video_indices = (
            batch["video"],
            batch["label"],
            batch["video_index"].type(torch.int32)
        )
        captions = list(self.index_to_caption.values())
        logits_per_video, logits_per_text = self(frames, captions)
        self.retrieval_recall.update(logits_per_video, video_indices)

    def test_epoch_end(self, outputs) -> None:
        r1_tot, r5_tot, r10_tot = self.retrieval_recall.compute()
        self.log("r1", r1_tot)
        self.log("r5", r5_tot)
        self.log("r10", r10_tot)

    def validation_epoch_end(self, outputs) -> None:
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

        self.log('val_top1_accuracy_temporal', temporal_top1_acc)
        self.log('val_top1_accuracy_static', static_top1_acc)

        self.log('val_top5_accuracy_temporal', temporal_top5_acc)
        self.log('val_top5_accuracy_static', static_top5_acc)

        self.log('val_top1_accuracy_total', self.top1_accuracy)
        self.log('val_top5_accuracy_total', self.top5_accuracy)

    def _encode_text(self, captions: List[str]) -> torch.Tensor:
        tokenized_captions = clip.tokenize(captions).to(self.device)
        eot_indices = (tokenized_captions
                       == self.eot_token).nonzero(as_tuple=True)[1]

        x = self.clip_model.token_embedding(tokenized_captions)

        if self.textual_context_addition:
            x, eot_indices = self.textual_context_addition(
                x,
                eot_indices,
                captions,
            )

        x = self._modified_text_encode(x, eot_indices)

        return x

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

    def _encode_image(self, video):
        if video.dim == 4:
            video = video.unsqueeze(0)
        pred_frames = video[:, :, self.pred_frames]
        # to_image = torchvision.transforms.ToPILImage()
        # image = to_image(pred_frames[0].squeeze())
        if self.visual_context_addition:
            visual_context = self.visual_context_addition(video)
            if self.vca_permutation:
                pred_frames, visual_context = self.vca_permutation(
                    pred_frames,
                    visual_context
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
        batch_size, channels, nr_frames, height, width = x.shape
        x = torch.permute(x, dims=(0, 2, 1, 3, 4))
        x = self.clip_model.visual.conv1(x.reshape(-1, channels, height, width))
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
                                     self.visual_context_addition.nr_output_vectors,
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

        self.pred_frames = [round(frame_idx) for frame_idx in pred_frames]

    def _freeze_components(self) -> None:
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def on_test_start(self) -> None:
        self._create_test_metrics()
        if self.visual_context_addition:
            self.visual_context_addition.set_val_test_transforms()
        self.index_to_caption = self.trainer.datamodule.index_to_caption

    def on_fit_start(self) -> None:
        self.index_to_caption = self.trainer.datamodule.index_to_caption

    def on_validation_start(self) -> None:
        if self.visual_context_addition:
            self.visual_context_addition.set_val_test_transforms()

    def on_train_start(self) -> None:
        if self.visual_context_addition:
            self.visual_context_addition.set_train_transforms()
