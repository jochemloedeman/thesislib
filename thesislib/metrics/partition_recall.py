import numpy as np
import torch
from scipy.special import softmax
from torchmetrics import Metric


class PartitionRecall(Metric):
    def __init__(self, partition):
        super().__init__()
        self.partition = partition
        self.add_state("similarities", default=[])
        self.txt2vis = None

    def update(self, batch_similarity):
        self.similarities.append(batch_similarity)

    def compute(self):
        similarity_matrix = torch.cat(self.similarities).cpu().numpy()
        image_ranks = self._calculate_image_ranks(similarity_matrix)
        return self._compute_metrics(image_ranks)

    @torch.no_grad()
    def _compute_metrics(self, image_ranks):
        eval_dict = self._calculate_recalls(image_ranks,
                                            self.partition.dynamic['cap_ids'],
                                            self.partition.static['cap_ids'])

        return eval_dict

    def _calculate_image_ranks(self, sim_matrix):
        probs = softmax(sim_matrix, axis=1)
        image_ranks = np.zeros(probs.T.shape[0])

        for index, score in enumerate(probs.T):
            inds = np.argsort(score)[::-1]
            image_ranks[index] = np.where(inds == self.txt2vis[index])[0][0]

        return image_ranks

    @staticmethod
    def _calculate_recalls(ranks, dynamic_ids, static_ids):
        dynamic_ranks = ranks[dynamic_ids]
        static_ranks = ranks[static_ids]

        r1_tot = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5_tot = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10_tot = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        if len(dynamic_ranks > 0):
            r1_dyn = 100.0 * len(np.where(dynamic_ranks < 1)[0]) / len(dynamic_ranks)
            r5_dyn = 100.0 * len(np.where(dynamic_ranks < 5)[0]) / len(dynamic_ranks)
            r10_dyn = 100.0 * len(np.where(dynamic_ranks < 10)[0]) / len(dynamic_ranks)
        else:
            r1_dyn, r5_dyn, r10_dyn = 0, 0, 0

        if len(static_ranks > 0):
            r1_stat = 100.0 * len(np.where(static_ranks < 1)[0]) / len(static_ranks)
            r5_stat = 100.0 * len(np.where(static_ranks < 5)[0]) / len(static_ranks)
            r10_stat = 100.0 * len(np.where(static_ranks < 10)[0]) / len(static_ranks)
        else:
            r1_stat, r5_stat, r10_stat = 0, 0, 0

        return {'static': (r1_stat, r5_stat, r10_stat),
                'dynamic': (r1_dyn, r5_dyn, r10_dyn),
                'total': (r1_tot, r5_tot, r10_tot)}
