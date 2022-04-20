import json
from pathlib import Path

if __name__ == '__main__':
    split = 'train'
    with open(Path(__file__).parent.parent / 'data' / 'coco' / f'captions_{split}2014.json') as annot_file:
        annots = json.load(annot_file)

    new_annots = {}
    for image_dict in annots['images']:
        image_path = f"{split}2014/{image_dict['file_name']}"
        image_id = image_dict['id']
        new_annots[image_id] = {'image': image_path, 'caption': []}

    for caption_dict in annots['annotations']:
        image_id = caption_dict['image_id']
        new_annots[image_id]['caption'].append(caption_dict['caption'])

    list_annotations = [value for key, value in new_annots.items()]

    with open(Path(__file__).parent.parent / 'data' / 'coco' / f'{split}_2014.json', 'w') as new_annot_file:
        json.dump(list_annotations, new_annot_file)


    print()