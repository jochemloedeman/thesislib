import csv
import json
from pathlib import Path

if __name__ == '__main__':
    new_annots = []
    with open(Path(__file__).parent.parent.parent / 'data' / 'msrvtt' / 'MSRVTT_JSFUSION_test.csv', newline='') as csvfile:
        annots = csv.reader(csvfile, delimiter=',')
        next(annots)
        for row in annots:
            _, _, video_id, caption = row
            new_annots.append({'video': f'test_videos/{video_id}.mp4', 'caption': [caption]})

    with open(Path(__file__).parent.parent.parent / 'data' / 'msrvtt' / f'test_jsfusion_msrvtt.json', 'w') as new_annot_file:
        json.dump(new_annots, new_annot_file)

