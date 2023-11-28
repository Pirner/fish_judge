from dataclasses import dataclass


@dataclass
class DataPoint:
    """
    data point for training, means it's an annotated data point
    """
    im_path: str
    label_id: int
