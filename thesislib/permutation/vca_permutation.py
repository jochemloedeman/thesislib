import torch


class VCAPermutation(torch.nn.Module):
    def __init__(self, permutation_mode):
        super().__init__()
        self.permutation_mode = permutation_mode

    def forward(self, frames: torch.Tensor, context: torch.Tensor):
        batch_size = len(frames)
        assert context.shape[0] == batch_size
        permutation = torch.randperm(batch_size)
        if self.permutation_mode == 'frames':
            return frames[permutation], context
        elif self.permutation_mode == 'context':
            return frames, context[permutation]
        else:
            return frames, context
