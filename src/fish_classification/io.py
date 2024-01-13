import cv2
import numpy as np
from torchvision import transforms as T
from PIL import Image

from src.fish_classification.transforms import FishTransforms


class FishClassificationIO:
    @staticmethod
    def load_im_from_disk(im_path: str, im_h: int, im_w: int) -> np.ndarray:
        """
        load an image from disk for inference
        :param im_path: image path
        :param im_h: image height
        :param im_w: image width
        :return:
        """
        transforms = FishTransforms.get_train_transforms(im_h=im_w, im_w=im_w)
        img = cv2.imread(im_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        aug = transforms(image=img)
        img = Image.fromarray(aug['image'])

        t = T.Compose([T.ToTensor(), ])
        img = t(img)
        return img
