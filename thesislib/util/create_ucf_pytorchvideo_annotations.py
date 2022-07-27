import argparse
import csv

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
    path_file_path = Path("/home/jochem/Documents/ai/scriptie/data/ucf101") / "annotations" / "testlist01.txt"
    labels_to_id_path = Path(
        "/home/jochem/Documents/ai/scriptie/data/ucf101") / "labels_to_id.csv"
    output_file_path = Path("/home/jochem/Documents/ai/scriptie/data/ucf101") / "annotations" / "test.csv"
    labels_to_id = pd.read_csv(labels_to_id_path).to_dict()['name']
    id_to_labels = {value: key for key, value in labels_to_id.items()}

    with open(path_file_path, "r") as file:
        paths = file.read().splitlines()
        with open(output_file_path, "w") as output_file:
            writer = csv.writer(output_file, delimiter=" ")
            for path in paths:
                class_id = str(id_to_labels[path.split("/")[0]])
                writer.writerow([path, class_id])



