import argparse
import csv
import json
from pathlib import Path

import pandas as pd

split_to_path = {
    "train": "train_val_annotation/train_val_videodatainfo.json",
    "val": "train_val_annotation/train_val_videodatainfo.json",
    "test": "test_videodatainfo.json"
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test', type=str)
    args = parser.parse_args()
    file_path = Path("/home/jochem/Documents/ai/scriptie/data") / 'msrvtt' / "annotations" / "MSRVTT_JSFUSION_test.csv"
    annotations = pd.read_csv(file_path).to_dict()
    list_annotations = []
    with open(Path(file_path).parent / f"{args.split}.csv", "w") as file:
        for idx in annotations["video_id"]:
            file.write(annotations["video_id"][idx] + " idx\n")