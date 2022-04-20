import json
import os

import torchvision
from torch.utils.data import Dataset

from thesislib.frame_extractor import FrameExtractor
from thesislib.util import pre_caption


class MSRVTTFrames(Dataset):
    def __init__(self, video_root, ann_root, split, sample_strategy, sample_rate,
                 transform=torchvision.transforms.Compose([]), max_words=30):

        filenames = {'test': 'test_msrvtt.json', 'test_jsfusion': 'test_jsfusion_msrvtt.json'}

        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.transform = transform
        self.video_root = video_root
        self.sample_strategy = sample_strategy
        self.sample_rate = sample_rate

        self.text = []
        self.video = []
        self.txt2vis = {}
        self.vis2txt = {}

        self.frame_extractor = FrameExtractor(strategy=self.sample_strategy, sample_rate=self.sample_rate)

        txt_id = 0
        for vid_id, ann in enumerate(self.annotation):
            self.video.append(ann['video'])
            self.vis2txt[vid_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, max_words))
                self.vis2txt[vid_id].append(txt_id)
                self.txt2vis[txt_id] = vid_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        video_path = os.path.join(self.video_root, self.annotation[index]['video'])
        frames = self.frame_extractor.extract(video_path=video_path)
        frames = None if frames is None else self.transform(frames)

        return frames, index, video_path
