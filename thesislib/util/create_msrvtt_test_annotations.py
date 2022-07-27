import collections
import csv
import json
import random

import pandas as pd

if __name__ == '__main__':
    test_paths = "/home/jochem/Documents/ai/scriptie/data/msrvtt/annotations/test.csv"
    caption_path = "/home/jochem/Documents/ai/scriptie/data/msrvtt/annotations/test_captions.csv"
    data_path = "/home/jochem/Documents/ai/scriptie/data/msrvtt/annotations/MSRVTT_JSFUSION_test.csv"

    annotations = pd.read_csv(data_path)
    sentence_by_fn = collections.defaultdict(list)
    for row in annotations.iterrows():
        sentence_by_fn[row[1]['video_id']].append(row[1]['sentence'])

    path_by_id = pd.read_csv(data_path).to_dict()['video_id']

    with open(test_paths, "w") as test_file:
        writer = csv.writer(test_file, delimiter=" ")
        for key, value in path_by_id.items():
            writer.writerow([value + ".mp4", key])
    with open(caption_path, "w") as caption_file:
        header = ["video_id", "sentence"]
        writer = csv.writer(caption_file, delimiter=",")
        writer.writerow(header)
        for vid_id, vid_fn in path_by_id.items():
            for sentence in sentence_by_fn[vid_fn]:
                writer.writerow([vid_id, sentence])