from typing import List

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
from PIL import Image
import cv2

from src.fish_classification.DTO import DataPoint


class FishClassificationDataset(Dataset):
    def __init__(
            self,
            data: List[DataPoint],
            n_classes: int,
            mean=0,
            std=1,
            transform=None
    ):
        """
        dataset for water segmentation in flooded areas
        :param data: data points to create train classifier from
        :param n_classes: number of classes
        :param mean: mean value for standardization
        :param std: standard deviation
        :param transform: transformation to run
        """
        self.data_points = data
        self.mean = mean
        self.std = std
        self.n_classes = n_classes

        self.transform = transform

    def __len__(self):
        """
        length of the dataset
        :return:
        """
        return len(self.data_points)

    def __getitem__(self, idx):
        """
        get an item from the data loading
        :param idx: index to load item from
        :return:
        """
        data_point = self.data_points[idx]

        img = cv2.imread(data_point.im_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = np.zeros(shape=self.n_classes, dtype=int)
        label[data_point.label_id] = 1

        if self.transform is not None:
            aug = self.transform(image=img)
            img = aug['image']

        if self.transform is None:
            img = Image.fromarray(img)

        # t = T.Compose([T.ToTensor(), ])
        # img = t(img)
        label = torch.from_numpy(label).long()

        label = label.type(torch.FloatTensor)

        return img, label
