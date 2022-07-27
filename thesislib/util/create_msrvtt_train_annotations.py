import collections
import csv
import json
import random

import pandas as pd

if __name__ == '__main__':
    train_val_paths = "/home/jochem/Documents/ai/scriptie/data/msrvtt/annotations/MSRVTT_train.9k.csv"
    train_paths_path = "/home/jochem/Documents/ai/scriptie/data/msrvtt/annotations/train.csv"
    val_paths_path = "/home/jochem/Documents/ai/scriptie/data/msrvtt/annotations/val.csv"
    caption_path = "/home/jochem/Documents/ai/scriptie/data/msrvtt/annotations/trainval_captions.csv"
    data_path = "/home/jochem/Documents/ai/scriptie/data/msrvtt/annotations/MSRVTT_data.json"

    sentences = json.load(open(data_path, "r"))['sentences']
    sentence_by_fn = collections.defaultdict(list)
    for sentence in sentences:
        sentence_by_fn[sentence['video_id']].append(sentence)

    path_by_id = pd.read_csv(train_val_paths).to_dict()['video_id']

    val_ids = random.sample(sorted(path_by_id), k=500)
    train_paths = {key: value for key, value in path_by_id.items() if key not in val_ids}
    val_paths = {key: value for key, value in path_by_id.items() if key in val_ids}
    with open(train_paths_path, "w") as train_file:
        writer = csv.writer(train_file, delimiter=" ")
        for key, value in train_paths.items():
            writer.writerow([value + ".mp4", key])
    with open(val_paths_path, "w") as val_file:
        writer = csv.writer(val_file, delimiter=" ")
        for key, value in val_paths.items():
            writer.writerow([value + ".mp4", key])
    with open(caption_path, "w") as caption_file:
        header = ["video_id", "sentence"]
        writer = csv.writer(caption_file, delimiter=",")
        writer.writerow(header)
        for vid_id, vid_fn in path_by_id.items():
            for sentence_dict in sentence_by_fn[vid_fn]:
                writer.writerow([vid_id, sentence_dict['caption']])