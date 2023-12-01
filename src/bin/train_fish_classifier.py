import pytorch_lightning as pl

from src.fish_classification.data import DataUtils
from src.fish_classification.model_factory import ConvNextTiny
from src.fish_classification.training import FishClassificationModule
from src.modeling.custom_callbacks import TraceSavingCB


def main():
    im_h = 224
    im_w = 224
    dataset_path = r'C:\data\fish_judge\fish_data\Fish_Dataset\Fish_Dataset'
    train_loader, val_loader = DataUtils.create_generators(
        dataset_path,
        im_h=im_h,
        im_w=im_w,
    )
    model = ConvNextTiny(num_classes=DataUtils.get_n_classes_from_dataset(dataset_path))

    model = FishClassificationModule(model=model, n_classes=DataUtils.get_n_classes_from_dataset(dataset_path))
    print('Started Training')
    trainer = pl.Trainer(accelerator='gpu',
                         max_epochs=6,
                         callbacks=[
                             TraceSavingCB(im_h=im_h, im_w=im_w, device='cuda'),
                         ])

    trainer.fit(model, train_loader, val_loader)
    print('Finished Training')


if __name__ == '__main__':
    main()
