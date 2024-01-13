import os
import glob

from tqdm import tqdm
import torch

from src.fish_classification.io import FishClassificationIO


def main():
    data_path = r'C:\data\fish_judge\test_data'
    model_path = r'C:\data\fish_judge\models\traced_model.pth'
    model = torch.jit.load(model_path)
    model.cpu()
    im_paths = glob.glob(os.path.join(data_path, '**/*.jpg'), recursive=True)
    for im_p in tqdm(im_paths):
        x_in = FishClassificationIO.load_im_from_disk(im_p, 224, 224)
        x_in = torch.unsqueeze(x_in, dim=0)
        y = model(x_in)
        exit(0)


if __name__ == '__main__':
    main()
