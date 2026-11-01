import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHS = ('resnet18', 'resnet34', 'efficientnet_b0')

# Parameter-name prefixes that make up the "last blocks" unfrozen in Stage 2.
# EfficientNet-B0's features[6] and features[7] are its last two MBConv stages,
# which play the same role as layer3/layer4 in ResNet.
_LAST_BLOCKS = {
    'resnet18':        ('layer3', 'layer4', 'fc'),
    'resnet34':        ('layer3', 'layer4', 'fc'),
    'efficientnet_b0': ('features.6', 'features.7', 'features.8', 'classifier'),
}

_HEAD = {
    'resnet18':        'fc',
    'resnet34':        'fc',
    'efficientnet_b0': 'classifier',
}


def build_model(num_classes, freeze_backbone=True, arch='resnet18'):
    if arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(512, num_classes)
    elif arch == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(512, num_classes)
    elif arch == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch '{arch}', expected one of {SUPPORTED_ARCHS}")

    if freeze_backbone:
        head = _HEAD[arch]
        for name, param in model.named_parameters():
            if not name.startswith(head):
                param.requires_grad = False

    return model


def unfreeze_last_blocks(model, arch='resnet18'):
    prefixes = _LAST_BLOCKS[arch]
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(prefixes)


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def gradcam_target_layer(model, arch='resnet18'):
    if arch.startswith('resnet'):
        return model.layer4[-1]
    return model.features[-1]


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
