import json
from pathlib import Path

if __name__ == '__main__':
    with open(Path(__file__).parent.parent / 'data' / 'msrvtt' / 'test_videodatainfo.json', "r") as annot_file:
        annots = json.load(annot_file)

    videos, sentences = annots['videos'], annots['sentences']
    new_annotations = {}
    for video in videos:
        video_id = video['video_id']
        file_path = f"test_videos/{video_id}.mp4"
        caption = []
        new_annotations[video_id] = {"video": file_path, "caption": caption}

    for sentence in sentences:
        video_id = sentence["video_id"]
        new_annotations[video_id]["caption"].append(sentence["caption"])

    list_annotations = [value for key, value in new_annotations.items()]

    with open(Path(__file__).parent.parent / 'data' / 'msrvtt' / 'test_msrvtt.json', "w") as new_annot_file:
        json.dump(list_annotations, new_annot_file)