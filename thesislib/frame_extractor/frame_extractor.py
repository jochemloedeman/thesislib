import av
import decord
import numpy as np
import torch
from PIL import Image
from torchvision.io import read_video
from decord import VideoReader


class FrameExtractor:

    def extract(self, video_path):
        return self._extract_decord(video_path)

    @staticmethod
    def _extract_decord(video_path):
        reader = VideoReader(
            uri=video_path,
            ctx=decord.cpu()
        )
        video_tensor = torch.permute(
            torch.tensor(reader[len(reader) // 2].asnumpy()),
            dims=(2, 0, 1)
        )
        if len(video_tensor) == 3:
            video_tensor = video_tensor.unsqueeze(0)
        return video_tensor

    # @staticmethod
    # def _extract_pyav(video_path):
    #
    #     container = av.open(video_path)
    #     stream = container.streams.video[0]
    #     container.seek(400, stream=stream)
    #     for idx, frame in enumerate(container.decode(stream)):
    #         frame_array = frame.to_ndarray(format='rgb24')
    #         im = Image.fromarray(frame_array)
    #         im.show()
    #     # video_tensor = torch.permute(
    #     #     torch.tensor(frame_array),
    #     #     dims=(2, 0, 1)
    #     # )
    #     # if len(video_tensor) == 3:
    #     #     video_tensor = video_tensor.unsqueeze(0)
    #     # return video_tensor

# if __name__ == '__main__':
