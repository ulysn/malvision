import torch.nn as nn
from torchvision import models


def build_model(num_classes, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    return model


def unfreeze_last_blocks(model):
    for name, param in model.named_parameters():
        param.requires_grad = any(s in name for s in ('layer3', 'layer4', 'fc'))


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
