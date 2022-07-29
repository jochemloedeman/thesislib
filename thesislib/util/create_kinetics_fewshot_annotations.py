import argparse
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', default=0.01, type=float)
    parser.add_argument('--data_path', default="/home/jochem/Documents/ai/scriptie/data", type=str)
    args = parser.parse_args()

    train_annot_path = f"{args.data_path}/kinetics/annotations/train.csv"
    new_train_annot_path = f"{args.data_path}/kinetics/annotations/train_fewshot.csv"
    val_annot_path = f"{args.data_path}/kinetics/annotations/validate.csv"
    new_val_annot_path = f"{args.data_path}/kinetics/annotations/validate_fewshot.csv"
    index_path = f"{args.data_path}/kinetics/annotations/labels_to_id.csv"

    train_lines = open(train_annot_path).read().splitlines()
    val_lines = open(val_annot_path).read().splitlines()
    new_nr_train_lines = int(len(train_lines) * args.fraction)
    new_nr_val_lines = int(len(val_lines) * args.fraction)
    new_train_lines = random.sample(train_lines, k=new_nr_train_lines)
    new_val_lines = random.sample(val_lines, k=new_nr_val_lines)

    with open(new_train_annot_path, "w") as out_file:
        out_file.writelines(line + "\n" for line in new_train_lines)

    with open(new_val_annot_path, "w") as out_file:
        out_file.writelines(line + "\n" for line in new_val_lines)
