import torch.nn as nn
import torchvision.models as models

RESNET_VERSIONS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}

def build_model(
    num_classes = 5,
    resnet_version = "resnet18",
    freeze_backbone = True,
    unfreeze_last_layers = 1
):

    # model = models.resnet18(
    #     weights="IMAGENET1K_V1"
    # )

    model_fn = RESNET_VERSIONS[resnet_version]
    model = model_fn(weights="IMAGENET1K_V1")

    if freeze_backbone:

        for param in model.parameters():
            param.requires_grad = False

        if unfreeze_last_layers > 0:

            layers_to_unfreeze = []
            all_layers = ["layer4", "layer3", "layer2", "layer1"]

            for i in range(min(unfreeze_last_layers, len(all_layers))):
                layers_to_unfreeze.append(all_layers[i])

            for layer_name in layers_to_unfreeze:
                if hasattr(model, layer_name):
                    for param in getattr(model, layer_name).parameters():
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
    model = build_model(num_classes=5, resnet_version="resnet50", freeze_backbone=True, unfreeze_last_layers=1)
    trainable, total = count_trainable_params(model)

    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")