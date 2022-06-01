from pathlib import Path

from torchvision import datasets

if __name__ == '__main__':
    root_dir = '/project/prjsloedeman/data'
    # root_dir = (Path(__file__).parents[3] / 'data').as_posix()
    kinetics400_train = datasets.Kinetics(
        root=root_dir,
        frames_per_clip=2,
        num_classes='400',
        num_workers=4,
        num_download_workers=4,
        # download=True,
        split='train'
    )
    kinetics400_val = datasets.Kinetics(
        root=root_dir,
        frames_per_clip=2,
        num_classes='400',
        num_workers=4,
        num_download_workers=4,
        # download=True,
        split='val'
    )