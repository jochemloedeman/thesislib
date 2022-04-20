from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer
from nltk.corpus.reader import WordNetError

from thesislib.partition import DynamicMetadata
from thesislib.partition import Partition
from thesislib.util import pre_caption


class Partitioner:
    def __init__(self, annotation, caption_threshold, dynamic_lexnames, modality, wsd=None):
        self.annotation = annotation
        self.caption_threshold = caption_threshold
        self.dynamic_lexnames = dynamic_lexnames
        self.modality = modality
        self.wsd = wsd

    def _create_indexes(self):
        txt_id = 0
        text = []
        video = []
        txt2vis = {}
        vis2txt = {}
        for vid_id, ann in enumerate(self.annotation):
            video.append(ann[self.modality])
            vis2txt[vid_id] = []
            for i, caption in enumerate(ann['caption']):
                text.append(pre_caption(caption, 77))
                vis2txt[vid_id].append(txt_id)
                txt2vis[txt_id] = vid_id
                txt_id += 1

        return text, video, txt2vis, vis2txt

    def create_partition(self):
        dynamic_img_ids, static_img_ids = [], []
        dynamic_cap_ids, static_cap_ids = [], []
        dynamic_metadata, static_metadata = [], []
        text, video, txt2vis, vis2txt = self._create_indexes()
        for index, annot_dict in enumerate(tqdm(self.annotation)):
            caption_indices = vis2txt[index]
            captions = annot_dict['caption']
            dynamic, metadata = self.is_dynamic(captions)
            if dynamic:
                dynamic_img_ids += [index]
                dynamic_cap_ids += caption_indices
                dynamic_metadata += [metadata]
            else:
                static_img_ids += [index]
                static_cap_ids += caption_indices
                static_metadata += [metadata]

        return Partition(dynamic_vis_ids=dynamic_img_ids, static_vis_ids=static_img_ids,
                         dynamic_cap_ids=dynamic_cap_ids, static_cap_ids=static_cap_ids,
                         dynamic_metadata=dynamic_metadata, static_metadata=static_metadata)

    def is_dynamic(self, captions):
        if self.wsd is None:
            return self.is_dynamic_baseline(captions)
        else:
            return self.is_dynamic_wsd(captions)

    def is_dynamic_baseline(self, captions):
        lemmatizer = WordNetLemmatizer()
        absolute_threshold = int(self.caption_threshold * len(captions))
        dynamic_captions = 0
        dynamic_tokens, dynamic_lexnames, static_tokens = [], [], []
        for caption in captions:
            tokenized_caption = nltk.word_tokenize(caption)
            tagged_caption = nltk.pos_tag(tokenized_caption)
            for token in tagged_caption:
                if is_verb(token):
                    lem_token = lemmatizer.lemmatize(token[0], pos=wn.VERB)
                    try:
                        synset = wn.synset(f"{lem_token}.v.01")
                    except WordNetError:
                        break
                    if synset._lexname in self.dynamic_lexnames:
                        dynamic_tokens.append(lem_token)
                        dynamic_lexnames.append(str(synset._lexname))
                        dynamic_captions += 1
                    else:
                        static_tokens.append(lem_token)

        metadata = DynamicMetadata(dynamic_count=dynamic_captions, dynamic_verbs=dynamic_tokens,
                                   static_verbs=static_tokens, lexnames=dynamic_lexnames)
        return dynamic_captions >= absolute_threshold, metadata

    def is_dynamic_wsd(self, captions):
        pass


def is_verb(token):
    return True if token[1][0] == 'V' else False
