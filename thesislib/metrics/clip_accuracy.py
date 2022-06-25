from collections import defaultdict
from typing import Optional

import torch
import torchmetrics

class ClipAccuracy:
    def __init__(
            self,
            top_k: Optional[int] = None,
            num_classes: Optional[int] = None,
            average: Optional[str] = "micro"
    ) -> None:

        self.metric = torchmetrics.Accuracy(
            num_classes=num_classes,
            average=average,
            top_k=top_k
        )
        self.preds = []
        self.labels = []
        self.video_indices = []

    def update(self, preds, labels, indices) -> None:
        self.preds += preds.unsqueeze(1)
        self.labels += labels.unsqueeze(1)
        self.video_indices += indices.unsqueeze(1)

    def compute(self):
        self.metric.to(self.preds[0].device)
        total_preds = torch.cat(self.preds, dim=0).squeeze()
        total_labels = torch.cat(self.labels, dim=0).squeeze()
        total_indices = torch.cat(self.video_indices, dim=0).squeeze()

        aggregated_preds = self._aggregate_preds(total_preds,
                                                 total_indices)
        probs = aggregated_preds.softmax(dim=-1)
        aggregated_labels = self._aggregate_labels(total_labels,
                                                  total_indices)

        self.metric.update(probs, aggregated_labels)
        return self.metric.compute()

    def __call__(self, preds, labels, indices):
        self.update(preds, labels, indices)

    @staticmethod
    def _aggregate_preds(tensors, indices):
        inv_index = defaultdict(list)
        aggregated_tensors = []
        for i, idx in enumerate(indices):
            idx = idx.item()
            inv_index[idx] += [i]
        for idx in inv_index:
            aggregated_tensors.append(
                tensors[inv_index[idx]].mean(dim=0)
            )

        return torch.stack(aggregated_tensors)

    @staticmethod
    def _aggregate_labels(tensors, indices):
        inv_index = defaultdict(list)
        aggregated_tensors = []
        for i, idx in enumerate(indices):
            idx = idx.item()
            inv_index[idx] += [i]
        for idx in inv_index:
            aggregated_tensors.append(tensors[inv_index[idx][0]])

        return torch.stack(aggregated_tensors)

