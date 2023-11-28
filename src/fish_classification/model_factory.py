import torch.nn as nn
from timm import create_model


class ConvNextTiny(nn.Module):
    def __init__(self, num_classes: int, dropout=0.1):
        super().__init__()
        self.model_name = 'convnext_tiny'
        self.backbone = create_model(self.model_name, pretrained=True, num_classes=num_classes)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(num_classes)
        self.sm = nn.Softmax(dim=1)

    def forward(self, sample):
        x = sample
        x = self.backbone(x)
        x = self.drop(x)
        logit = self.fc(x)
        logit = self.sm(logit)
        return logit
