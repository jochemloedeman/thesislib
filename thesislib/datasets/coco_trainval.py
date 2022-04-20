import os
import json

import torch
import torchvision
from torch.utils.data import Dataset

from PIL import Image
from thesislib.util.utils import pre_caption


class COCOCaptions(Dataset):
    def __init__(self,
                 image_root,
                 ann_root,
                 split,
                 partition,
                 transform=torchvision.transforms.Compose([]),
                 max_words=30):
        """
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        filenames = {'train': 'train_2014.json', 'val': 'val_2014.json'}
        self.split = split
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.dynamic_ids = set(partition.dynamic['vis_ids'])

        self.captions_per_image = 5
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

        self.text_length = txt_id

    def __len__(self):
        return self.text_length if self.split == 'train' else len(self.annotation)

    def __getitem__(self, index):

        if self.split == 'train':
            img_id = self.txt2vis[index]
            txt_idx = self.vis2txt[img_id].index(index)
        else:
            img_id = index
            txt_idx = 0

        image_path = os.path.join(self.image_root, self.annotation[img_id]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = [self.text[text_idx] for text_idx in self.vis2txt[img_id]][txt_idx]
        is_dynamic = img_id in self.dynamic_ids
        return image, img_id, caption, is_dynamic


def caption_collate(batch):
    images = torch.stack([element[0] for element in batch])
    captions = [element[2] for element in batch]
    dyn_bools = torch.tensor([element[3] for element in batch])
    return images, captions, dyn_bools
