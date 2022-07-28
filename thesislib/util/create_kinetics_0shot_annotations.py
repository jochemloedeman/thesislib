import argparse
import csv
import random

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=200, type=int)
    args = parser.parse_args()
    train_annot_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/train.csv"
    new_train_annot_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/train_0shot.csv"
    val_annot_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/validate.csv"
    new_val_annot_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/validate_0shot.csv"
    zeroshot_classes_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/0shot_classes.txt"
    index_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/labels_to_id.csv"
    new_index_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/labels_to_id_0shot.csv"

    zeroshot_classes = random.sample(range(400), k=args.k)
    train_classes = [index for index in range(400) if
                     index not in zeroshot_classes]
    index_map = {train_classes[index]: index
                 for index in range(len(train_classes))}

    open(zeroshot_classes_path, "w").writelines(
        [str(elem) + "\n" for elem in zeroshot_classes])
    with open(train_annot_path, "r") as in_file:
        reader = csv.reader(in_file, delimiter=" ")
        with open(new_train_annot_path, "w") as out_file:
            writer = csv.writer(out_file, delimiter=" ")
            for row in reader:
                if int(row[1]) not in zeroshot_classes:
                    writer.writerow([row[0], index_map[int(row[1])]])

    with open(val_annot_path, "r") as in_file:
        reader = csv.reader(in_file, delimiter=" ")
        with open(new_val_annot_path, "w") as out_file:
            writer = csv.writer(out_file, delimiter=" ")
            for row in reader:
                if int(row[1]) not in zeroshot_classes:
                    writer.writerow([row[0], index_map[int(row[1])]])

    labels = open(zeroshot_classes_path).read().splitlines()
    labels = [int(label) for label in labels]
    with open(new_index_path, "w") as out_file:
        writer = csv.writer(out_file, delimiter=",")
        writer.writerow(['id', 'name'])
        with open(index_path, "r") as in_file:
            index_df = pd.read_csv(in_file)
            for row in index_df.iterrows():
                if row[1]['id'] not in labels:
                    new_id = index_map[int(row[1]['id'])]
                    writer.writerow([new_id, row[1]['name']])
