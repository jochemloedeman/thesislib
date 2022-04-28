import os
import json
import torch
import torchvision
from clip import clip
from torch.utils.data import Dataset
from PIL import Image
from thesislib.util import pre_caption


class COCOCaptionsVal(Dataset):
    def __init__(self,
                 image_root,
                 ann_root,
                 split,
                 partition,
                 nr_text_chunks,
                 transform=torchvision.transforms.Compose([]),
                 max_words=30):
        """
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        filenames = {'val': 'val_2014.json', 'test': 'coco_karpathy_test.json'}
        self.split = split
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.dynamic_vis_ids = set(partition.dynamic['vis_ids'])
        self.dynamic_cap_ids = set(partition.dynamic['cap_ids'])
        self.nr_text_chunks = nr_text_chunks
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
        self.dyn_bools = self._create_dyn_bools()
        split_text, split_bools = self._split_texts_and_bools()
        self.tokenized_captions = [clip.tokenize(split) for split in split_text]
        self.split_bools = split_bools

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        is_dynamic = index in self.dynamic_vis_ids
        return image, index, is_dynamic

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


def val_test_collate(batch):
    images = torch.stack([element[0] for element in batch])
    indices = [element[1] for element in batch]
    dyn_bools = torch.tensor([element[2] for element in batch])
    return images, indices, dyn_bools
