import os
import json

import torchvision.transforms
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from thesislib.util.utils import pre_caption


class Flickr30kDataset(Dataset):
    def __init__(self, image_root, ann_root, split, transform=torchvision.transforms.Compose([]), max_words=30):
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
                'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
        filenames = {'val': 'flickr30k_val.json', 'test': 'flickr30k_test.json'}

        download_url(urls[split], ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.transform = transform
        self.image_root = image_root

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

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index, image_path
