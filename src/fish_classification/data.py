import os
import glob

from src.fish_classification.DTO import DataPoint


class DataUtils:
    @staticmethod
    def create_generators(dataset_path: str, val_size=0.2):
        """
        create train and validation generator from a given dataset
        :param dataset_path: dataset path to create generators from
        :param val_size: validation size for the dataset
        :return:
        """
        train_points = []
        val_points = []
        content = list(filter(lambda x: os.path.isdir(os.path.join(dataset_path, x)), os.listdir(dataset_path)))
        content = [os.path.join(dataset_path, x) for x in content]
        for i, class_path in enumerate(content):
            im_paths = list(glob.glob(os.path.join(class_path, '**/*.png'), recursive=True))
            n = int(len(im_paths) * val_size)
            val_paths = im_paths[:n]
            train_paths = im_paths[n:]
            train_points.extend([DataPoint(im_path=x, label_id=i) for x in train_paths])
            val_points.extend([DataPoint(im_path=x, label_id=i) for x in val_paths])
        exit(0)
