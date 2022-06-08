import math

import decord
import torch
from decord import VideoReader


class FrameSampler:

    def __init__(self, nr_frames=2):
        self.nr_frames = nr_frames

    def extract(self, video_path):
        return self._extract_decord(video_path)

    def _extract_decord(self, video_path):
        reader = VideoReader(
            uri=video_path,
            ctx=decord.cpu()
        )

        # fps = reader.get_avg_fps()
        video_length = len(reader)
        interval = video_length / (self.nr_frames + 1)
        indices = [(x+1)*interval for x in range(self.nr_frames)]
        indices = [math.floor(index) for index in indices]
        video_tensor = torch.permute(
            torch.tensor(reader.get_batch(indices).asnumpy()),
            dims=(0, 3, 1, 2)
        )
        return video_tensor
