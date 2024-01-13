import torch
from pytorch_lightning.callbacks import Callback


class TraceSavingCB(Callback):
    def __init__(self, im_h: int, im_w: int, device: str, model_path: str):
        """
        callback to save a traced model
        :param im_h:
        :param im_w:
        :param device:
        :param model_path: path to where storing the model during training
        """
        self.im_w = im_w
        self.im_h = im_h
        self.device = device
        self.model_path = model_path

    def on_train_start(self, trainer, pl_module):
        pass

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """

        :param trainer:
        :param pl_module:
        :return:
        """
        x = torch.rand(1, 3, self.im_h, self.im_w).to(self.device)
        traced_cell = torch.jit.trace(pl_module.model, x)
        traced_cell.save(self.model_path)

    def on_train_end(self, trainer, pl_module):
        """
        save the model
        :param trainer:
        :param pl_module:
        :return:
        """
        # Trace a specific method and construct `ScriptModule` with
        # a single `forward` method
        x = torch.rand(1, 3, self.im_h, self.im_w).to(self.device)
        traced_cell = torch.jit.trace(pl_module.model, x)
        traced_cell.save('my_module.pth')
