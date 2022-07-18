from collections import defaultdict

import numpy as np
import torch
from scipy.special import softmax
from torchmetrics import Metric


class RetrievalRecall(Metric):
    def __init__(self, txt2vis):
        super().__init__()
        self.add_state("preds", default=[])
        self.add_state("indices", default=[])
        self.txt2vis = txt2vis

    def update(self, preds, indices):
        self.preds.append(preds)
        self.indices.append(indices)

    def compute(self):
        total_preds = torch.cat(self.preds, dim=0).squeeze()
        total_indices = torch.cat(self.indices, dim=0).squeeze()
        aggregated_preds = self._aggregate_preds(total_preds,
                                                 total_indices)

        similarities = aggregated_preds.cpu().numpy()
        image_ranks = self._calculate_image_ranks(similarities)
        return self._compute_metrics(image_ranks)

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

    @torch.no_grad()
    def _compute_metrics(self, image_ranks):
        eval_dict = self._calculate_recalls(image_ranks)

        return eval_dict

    def _calculate_image_ranks(self, sim_matrix):
        probs = softmax(sim_matrix, axis=1)
        image_ranks = np.zeros(probs.T.shape[0])

        for index, score in enumerate(probs.T):
            inds = np.argsort(score)[::-1]
            image_ranks[index] = np.where(inds == self.txt2vis[index])[0][0]

        return image_ranks

    @staticmethod
    def _calculate_recalls(ranks):
        r1_tot = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5_tot = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10_tot = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        return r1_tot, r5_tot, r10_tot
