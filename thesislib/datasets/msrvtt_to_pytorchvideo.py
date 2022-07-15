from typing import Type, Optional, Callable, Dict, Any

import torch
from pytorchvideo.data import LabeledVideoDataset, ClipSampler, \
    labeled_video_dataset


def MSRVTT(
        data_path: str,
        clip_sampler: ClipSampler,
        video_sampler: Type[
            torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        video_path_prefix: str = "",
        decode_audio: bool = True,
        decoder: str = "pyav",
) -> LabeledVideoDataset:
    return labeled_video_dataset(
        data_path,
        clip_sampler,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder,
    )
