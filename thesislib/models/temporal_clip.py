from typing import Dict, Union, Optional, List

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision.transforms
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
from torch.nn.functional import cross_entropy

from ..components.tca import ConstantTCA, LMTCA
from ..components.vca import ConstantVCA, VideoVCA, ImageVCA
from ..metrics import ClipAccuracy
from ..permutation import VCAPermutation


class TemporalCLIP(pl.LightningModule):
    eot_token = SimpleTokenizer().encoder["<|endoftext|>"]

    def __init__(
            self,
            clip_architecture: str,
            nr_pred_frames: int,
            nr_context_frames: int,
            vca_settings: Union[Dict, None],
            tca_settings: Union[Dict, None],
            temporal_dataset: Dict,
            optimizer: str,
            lr_scheduler: Optional[str] = 'cosine',
            epochs: Optional[int] = 150,
            permutation_mode: Optional[str] = None,
            unseen_classes: Optional[List] = None,
    ) -> None:

        super().__init__()
        self.prompt_prefix = None
        self.save_hyperparameters()
        self.clip_model, _ = clip.load(clip_architecture, device='cpu')
        self.temporal_dataset = temporal_dataset
        self.nr_pred_frames = nr_pred_frames
        self.nr_context_frames = nr_context_frames
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.unseen_classes = unseen_classes
        self._get_pred_frames()
        self._build_permutation(permutation_mode)

        self.index_to_classes = None
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
        self.top5_accuracy = torchmetrics.Accuracy(
            top_k=5
        )
        self.top1_accuracy = torchmetrics.Accuracy(
            top_k=1
        )

    def _create_test_metrics(self, temporal_dataset):
        self.test_metrics = [
            ClipAccuracy(
                name='test_top1_accuracy',
                top_k=1,
            ),
            ClipAccuracy(
                name='test_top5_accuracy',
                top_k=5,
            ),
            ClipAccuracy(
                name='temporal_top1_accuracy',
                top_k=1,
                subset=temporal_dataset['temporal']
            ),
            ClipAccuracy(
                name='temporal_top5_accuracy',
                top_k=5,
                subset=temporal_dataset['temporal']
            ),
            ClipAccuracy(
                name='static_top1_accuracy',
                top_k=1,
                subset=temporal_dataset['static']
            ),
            ClipAccuracy(
                name='static_top5_accuracy',
                top_k=5,
                subset=temporal_dataset['static']
            ),
        ]

        if self.unseen_classes is not None:
            seen_classes = [
                idx for idx in range(self.temporal_dataset['number_of_classes'])
                if idx not in self.unseen_classes
            ]
            self.test_metrics.extend(
                [
                    ClipAccuracy(
                        name='unseen_class_top1_accuracy',
                        top_k=1,
                        subset=self.unseen_classes
                    ),
                    ClipAccuracy(
                        name='unseen_class_top5_accuracy',
                        top_k=5,
                        subset=self.unseen_classes
                    ),
                    ClipAccuracy(
                        name='seen_class_top1_accuracy',
                        top_k=1,
                        subset=seen_classes
                    ),
                    ClipAccuracy(
                        name='seen_class_top5_accuracy',
                        top_k=5,
                        subset=seen_classes
                    )
                ]
            )

    def forward(self, frames):
        video_features = self._encode_image(frames)
        video_features = video_features.mean(dim=1)
        video_features = video_features / video_features.norm(dim=1,
                                                              keepdim=True)
        text_features = self._encode_text()
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
        frames, labels = (
            batch["video"],
            batch["label"]
        )
        logits_per_video, logits_per_text = self(frames)
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

    def test_step(self, batch, batch_idx):
        frames, labels, video_indices = (
            batch["video"],
            batch["label"],
            batch["video_index"].type(torch.int32)
        )
        if isinstance(labels, list):
            labels = torch.tensor([self.class_to_id[label] for label in labels],
                                  device=self.device)

        logits_per_video, logits_per_text = self(frames)
        for metric in self.test_metrics:
            metric.update(logits_per_video, labels, video_indices)

    def test_epoch_end(self, outputs) -> None:
        for metric in self.test_metrics:
            self.log(metric.name, metric.compute())

    def validation_epoch_end(self, outputs) -> None:
        self.log('val_top1_accuracy_total', self.top1_accuracy)
        self.log('val_top5_accuracy_total', self.top5_accuracy)

    def _encode_text(self) -> torch.Tensor:
        eot_indices = (self.tokenized_prompts
                       == self.eot_token).nonzero(as_tuple=True)[1]

        x = self.clip_model.token_embedding(self.tokenized_prompts)

        if self.textual_context_addition:
            x, eot_indices = self.textual_context_addition(
                x,
                eot_indices,
                list(self.index_to_classes.values()),
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

    def _tokenize_prompts(self) -> None:
        class_prompts = list(self.index_to_classes.values())
        full_prompt = [self.prompt_prefix + " " + prompt for prompt in class_prompts]
        tokenized_prompts = clip.tokenize(full_prompt)
        self.tokenized_prompts = tokenized_prompts.to(self.device)

    def _freeze_components(self) -> None:
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def on_test_start(self) -> None:
        self._create_test_metrics(self.trainer.datamodule.temporal_dataset)
        self.prompt_prefix = self.trainer.datamodule.prompt_prefix
        if self.visual_context_addition:
            self.visual_context_addition.set_val_test_transforms()
        self.index_to_classes = self.trainer.datamodule.index_to_classes
        self.class_to_id = self.trainer.datamodule.class_to_id
        self._tokenize_prompts()

    def on_fit_start(self) -> None:
        self.prompt_prefix = self.trainer.datamodule.prompt_prefix
        self.index_to_classes = self.trainer.datamodule.index_to_classes
        self.class_to_id = self.trainer.datamodule.class_to_id
        self._tokenize_prompts()

    def on_validation_start(self) -> None:
        if self.visual_context_addition:
            self.visual_context_addition.set_val_test_transforms()

    def on_train_start(self) -> None:
        if self.visual_context_addition:
            self.visual_context_addition.set_train_transforms()
