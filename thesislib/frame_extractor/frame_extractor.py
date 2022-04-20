import torch
from torchvision.io import read_video


class FrameExtractor:

    def __init__(self, strategy, sample_rate=1):
        self.strategy = strategy
        self.sample_rate = sample_rate

    def extract(self, video_path):
        if self.strategy == 'average' or self.strategy == 'max_pool':
            return self._extract_uniform_frames(video_path)
        elif self.strategy == 'center':
            return self._extract_center_frame(video_path)
        elif self.strategy == 'full':
            return self._extract_all_frames(video_path)
        else:
            return None

    def _extract_uniform_frames(self, video_path):
        video_tensor, audio, md = read_video(video_path)
        sample_interval = int(md["video_fps"] / self.sample_rate)
        subsampled = torch.permute(video_tensor[::sample_interval], dims=(0, 3, 1, 2))
        subsampled = subsampled / 255
        return subsampled

    @staticmethod
    def _extract_center_frame(video_path):
        video_tensor, audio, md = read_video(video_path)
        nr_of_frames = len(video_tensor)
        subsampled = torch.permute(video_tensor[nr_of_frames // 2], dims=(2, 0, 1)).unsqueeze(0)
        subsampled = subsampled / 255
        return subsampled

    @staticmethod
    def _extract_all_frames(video_path):
        video_tensor, audio, md = read_video(video_path)
        subsampled = torch.permute(video_tensor, dims=(0, 3, 1, 2))
        subsampled = subsampled / 255
        return subsampled
