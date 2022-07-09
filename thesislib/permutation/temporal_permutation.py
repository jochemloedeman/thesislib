import torch


class TemporalPermutation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        nr_frames = frames.shape[2]
        permutation = torch.randperm(nr_frames)
        return frames[:, :, permutation, :, :]
