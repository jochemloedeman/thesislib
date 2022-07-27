import csv

import pandas

if __name__ == '__main__':
    labels_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/0shot_classes.txt"
    index_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/labels_to_id.csv"
    new_index_path = "/home/jochem/Documents/ai/scriptie/data/kinetics/annotations/labels_to_id_0shot.csv"
    labels = open(labels_path).read().splitlines()
    labels = [int(label) for label in labels]
    with open(new_index_path, "w") as out_file:
        writer = csv.writer(out_file, delimiter=",")
        writer.writerow(['id', 'name'])
        with open(index_path, "r") as in_file:
            index_df = pandas.read_csv(in_file)
            for row in index_df.iterrows():
                if row[1]['id'] not in labels:
                    writer.writerow([row[1]['id'], row[1]['name']])

    print()

