import os
import json

import torch
import torchvision
from clip import clip
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image
from thesislib.util import pre_caption


class COCOKarpathy(Dataset):
    def __init__(self,
                 image_root,
                 ann_root,
                 split,
                 partition,
                 nr_text_chunks,
                 transform=torchvision.transforms.Compose([]),
                 max_words=30):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val': 'coco_karpathy_val.json', 'test': 'coco_karpathy_test.json'}

        download_url(urls[split], ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.dynamic_vis_ids = set(partition.dynamic['vis_ids'])
        self.dynamic_cap_ids = set(partition.dynamic['cap_ids'])
        self.nr_text_chunks = nr_text_chunks

        self.text = []
        self.image = []
        self.txt2vis = {}
        self.vis2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.vis2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, max_words))
                self.vis2txt[img_id].append(txt_id)
                self.txt2vis[txt_id] = img_id
                txt_id += 1

        self.dyn_bools = self._create_dyn_bools()
        split_text, split_bools = self._split_texts_and_bools()
        self.tokenized_captions = [clip.tokenize(split) for split in split_text]
        self.split_bools = split_bools

    def _split_texts_and_bools(self):
        splitted_texts = []
        splitted_bools = []
        split_size = len(self.text) // self.nr_text_chunks
        for i in range(self.nr_text_chunks - 1):
            splitted_texts.append(self.text[(i * split_size):((i + 1) * split_size)])
            splitted_bools.append(self.dyn_bools[(i * split_size):((i + 1) * split_size)])
        splitted_texts.append(self.text[(split_size * (self.nr_text_chunks - 1)):])
        splitted_bools.append(self.dyn_bools[(split_size * (self.nr_text_chunks - 1)):])
        return splitted_texts, splitted_bools

    def _create_dyn_bools(self):
        text_indices = list(range(len(self.text)))
        dyn_bools = torch.tensor([text_index in self.dynamic_cap_ids for text_index in text_indices])
        return dyn_bools

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index