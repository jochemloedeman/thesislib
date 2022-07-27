import csv
import random

if __name__ == '__main__':
    train_annot_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/train.csv"
    new_train_annot_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/train_0shot.csv"
    val_annot_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/validate.csv"
    new_val_annot_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/validate_0shot.csv"
    class_list_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/0shot_classes.txt"
    zeroshot_classes = random.sample(range(400), k=50)
    zeroshot_set = set([str(elem) for elem in zeroshot_classes])
    open(class_list_path, "w").writelines([str(elem) + "\n" for elem in zeroshot_classes])
    with open(train_annot_path, "r") as in_file:
        reader = csv.reader(in_file, delimiter=" ")
        with open(new_train_annot_path, "w") as out_file:
            writer = csv.writer(out_file, delimiter=" ")
            for row in reader:
                if row[1] not in zeroshot_set:
                    writer.writerow(row)

    with open(val_annot_path, "r") as in_file:
        reader = csv.reader(in_file, delimiter=" ")
        with open(new_val_annot_path, "w") as out_file:
            writer = csv.writer(out_file, delimiter=" ")
            for row in reader:
                if row[1] not in zeroshot_set:
                    writer.writerow(row)

