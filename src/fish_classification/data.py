import os
import glob

import torch

from src.fish_classification.DTO import DataPoint
from src.fish_classification.generation import FishClassificationDataset
from src.fish_classification.transforms import FishTransforms


class DataUtils:
    @staticmethod
    def get_n_classes_from_dataset(dataset_path: str) -> int:
        """
        extract the number of datasets from the dataset given
        :param dataset_path: dataset to extract number of classes
        :return:
        """
        content = list(filter(lambda x: os.path.isdir(os.path.join(dataset_path, x)), os.listdir(dataset_path)))
        content = [os.path.join(dataset_path, x) for x in content]
        return len(content)

    @staticmethod
    def create_generators(
            dataset_path: str,
            im_h: int,
            im_w: int,
            val_size=0.2
    ):
        """
        create train and validation generator from a given dataset
        :param dataset_path: dataset path to create generators from
        :param im_h: image height
        :param im_w: image width
        :param val_size: validation size for the dataset
        :return:
        """
        train_points = []
        val_points = []
        content = list(filter(lambda x: os.path.isdir(os.path.join(dataset_path, x)), os.listdir(dataset_path)))
        content = [os.path.join(dataset_path, x) for x in content]
        for i, class_path in enumerate(content):
            im_paths_png = list(glob.glob(os.path.join(class_path, '**/*.png'), recursive=True))
            im_paths_jpg = list(glob.glob(os.path.join(class_path, '**/*.jpg'), recursive=True))
            im_paths = im_paths_jpg + im_paths_png
            im_paths = list(filter(lambda x: 'GT' not in x, im_paths))
            n = int(len(im_paths) * val_size)
            val_paths = im_paths[:n]
            train_paths = im_paths[n:]
            train_points.extend([DataPoint(im_path=x, label_id=i) for x in train_paths])
            val_points.extend([DataPoint(im_path=x, label_id=i) for x in val_paths])

        train_ds = FishClassificationDataset(
            data=train_points,
            n_classes=len(content),
            transform=FishTransforms.get_train_transforms(im_h=im_w, im_w=im_w),
        )
        val_ds = FishClassificationDataset(
            data=val_points,
            n_classes=len(content),
            transform=FishTransforms.get_val_transform(im_h=im_h, im_w=im_w),
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=2,
            shuffle=True,
            num_workers=1
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=2,
            num_workers=1
        )
        return train_loader, val_loader
