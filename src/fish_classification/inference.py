from PIL import Image
import numpy as np
import torch

from src import constants
from src.fish_classification.transforms import FishTransforms
from src.fish_classification.utils import FishClassificationUtils


class InferencePipeline:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.transforms = FishTransforms.get_val_transform(constants.im_h, constants.im_w)

    def classify_fish_name(self, im: np.ndarray) -> str:
        """
        classifies an image and extracts which fish is in there
        :param im: image to classify for fish
        :return:
        """
        model = torch.jit.load(self.model_path)
        model.cpu()
        channels_first_im = np.moveaxis(im, -1, 0)
        channels_first_im = torch.from_numpy(channels_first_im)
        x_in = self.transforms(channels_first_im)
        # x_in = data['image']
        x_in = torch.unsqueeze(x_in, dim=0)
        y = model(x_in)
        # convert prediction into string class
        prediction_result = FishClassificationUtils.convert_prediction_to_class(y[0])
        return prediction_result
