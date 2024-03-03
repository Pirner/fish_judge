import numpy as np

from src.constants import class_mapping


class FishClassificationUtils:
    @staticmethod
    def convert_prediction_to_class(prediction) -> str:
        """
        convert the prediction of a network into the class name itself
        :param prediction:
        :return:
        """
        result = prediction.cpu().detach().numpy()
        class_id = np.argmax(result, axis=0)
        class_name = class_mapping[class_id]
        return class_name
