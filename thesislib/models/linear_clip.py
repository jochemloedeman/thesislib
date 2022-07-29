from typing import Dict, Union, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
from torch.nn.functional import cross_entropy
from ..metrics import ClipAccuracy


class LinearCLIP(pl.LightningModule):
    eot_token = SimpleTokenizer().encoder["<|endoftext|>"]

    def __init__(
            self,
            clip_architecture: str,
            nr_pred_frames: int,
            nr_context_frames: int,
            temporal_dataset: Dict,
            optimizer: str,
            lr_scheduler: Optional[str] = 'cosine',
            epochs: Optional[int] = 150,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.clip_model, _ = clip.load(clip_architecture, device='cpu')
        self.linear_probe = torch.nn.Linear(
            in_features=self.clip_model.visual.output_dim,
            out_features=temporal_dataset['number_of_classes']
        )
        self.temporal_dataset = temporal_dataset
        self.nr_pred_frames = nr_pred_frames
        self.nr_context_frames = nr_context_frames
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self._get_pred_frames()

        self.index_to_classes = None
        self.class_to_id = None
        self._freeze_components()
        self._create_validation_metrics()

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

    def forward(self, frames):
        pred_frames = self._get_pred_frames()
        frames = frames[:, :, pred_frames]
        bs, channels, nr_frames, height, width = frames.shape
        frames = torch.permute(frames, dims=(0, 2, 1, 3, 4))
        video_features = self.clip_model.encode_image(
            frames.reshape(-1, channels, height, width)
        )
        video_features = video_features.reshape(bs, nr_frames, -1)
        video_features = video_features.mean(dim=1)
        logits = self.linear_probe(video_features)
        return logits


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
        logits_per_video = self.forward(frames)
        loss = cross_entropy(logits_per_video, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        frames, labels = (
            batch["video"],
            batch["label"]
        )
        logits_per_video = self(frames)
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

        logits_per_video = self(frames)
        for metric in self.test_metrics:
            metric.update(logits_per_video, labels, video_indices)

    def test_epoch_end(self, outputs) -> None:
        for metric in self.test_metrics:
            self.log(metric.name, metric.compute())

    def validation_epoch_end(self, outputs) -> None:
        self.log('val_top1_accuracy_total', self.top1_accuracy)
        self.log('val_top5_accuracy_total', self.top5_accuracy)

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

        return [round(frame_idx) for frame_idx in pred_frames]

    def _freeze_components(self) -> None:
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def on_test_start(self) -> None:
        self._create_test_metrics(self.trainer.datamodule.temporal_dataset)
        self.class_to_id = self.trainer.datamodule.class_to_id

    def on_fit_start(self) -> None:
        self.class_to_id = self.trainer.datamodule.class_to_id