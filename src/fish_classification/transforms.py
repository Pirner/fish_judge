import cv2
import albumentations as A


class FishTransforms:
    @staticmethod
    def get_train_transforms(im_h: int, im_w: int):
        """
        get training data transforms
        :param im_h: image height
        :param im_w: image width
        :return:
        """
        t_train = A.Compose([
            A.Resize(im_h, im_w, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
            A.GaussNoise()]
        )

        return t_train

    @staticmethod
    def get_val_transform(im_h: int, im_w: int):
        """
        get validation data transforms
        :param im_h: image height
        :param im_w: image width
        :return:
        """
        t_val = A.Compose([
            A.Resize(im_h, im_w, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.GridDistortion(p=0.2),
        ])

        return t_val
