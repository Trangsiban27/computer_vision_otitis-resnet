import torch.nn as nn
import torchvision.models as models

def build_model(
    num_classes = 5,
    freeze_backbone = True,
    unfreeze_last_block = True
):

    model = models.resnet18(
        weights="IMAGENET1K_V1"
    )

    if freeze_backbone:

        for param in model.parameters():
            param.requires_grad = False

        if unfreeze_last_block:

            for param in model.layer4.parameters():
                param.requires_grad = True

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model

def count_trainable_params(model):

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return trainable, total

if __name__ == '__main__':
    model = build_model(num_classes=5, freeze_backbone=True, unfreeze_last_block=True)
    trainable, total = count_trainable_params(model)

    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")