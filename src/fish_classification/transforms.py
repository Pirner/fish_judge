# import cv2
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import torch
from torchvision.transforms import v2


class FishTransforms:
    @staticmethod
    def get_train_transforms(im_h: int, im_w: int):
        """
        get training data transforms
        :param im_h: image height
        :param im_w: image width
        :return:
        """
        # t_train = A.Compose([
        #     A.Resize(im_h, im_w, interpolation=cv2.INTER_NEAREST),
        #     A.HorizontalFlip(),
        #     A.VerticalFlip(),
        #     A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
        #     A.GaussNoise(),
        #     A.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225],
        #     ),
        #     ToTensorV2(),
        # ])
        t_train = v2.Compose([
            v2.Resize(size=(im_h, im_w)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return t_train

    @staticmethod
    def get_val_transform(im_h: int, im_w: int):
        """
        get validation data transforms
        :param im_h: image height
        :param im_w: image width
        :return:
        """
        # t_val = A.Compose([
        #     A.Resize(im_h, im_w, interpolation=cv2.INTER_NEAREST),
        #     A.HorizontalFlip(),
        #     A.GridDistortion(p=0.2),
        #     A.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225],
        #     ),
        #     ToTensorV2(),
        # ])
        t_val = v2.Compose([
            v2.Resize(size=(im_h, im_w)),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return t_val
