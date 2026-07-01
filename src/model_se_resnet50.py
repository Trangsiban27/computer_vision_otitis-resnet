import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        squeeze = x.mean(dim=(2, 3))

        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)

        scale = excitation.view(b, c, 1, 1)
        return x * scale
    
class SEBottleneck(Bottleneck):
    # expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                groups=1, base_width=64, dilation=1, norm_layer=None,
                se_reduction=16):
        super().__init__(
            inplanes, planes, stride, downsample,
            groups, base_width, dilation, norm_layer
        )

        self.se = SEBlock(planes * self.expansion, reduction=se_reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
def _replace_bottleneck_with_se(module, se_reduction=16):
    for name, child in module.named_children():
        if isinstance(child, Bottleneck):
            inplanes = child.conv1.in_channels
            planes = child.conv1.out_channels  # đúng với Bottleneck (vì conv1 out = planes)
            stride = child.stride if hasattr(child, 'stride') else 1
            downsample = child.downsample
            groups = getattr(child, 'groups', 1)
            base_width = getattr(child, 'base_width', 64)
            dilation = getattr(child, 'dilation', 1)
            norm_layer = getattr(child, 'norm_layer', None)

            # se_block = SEBottleneck(
            #     child.conv1.in_channels,
            #     child.conv1.out_channels,
            #     stride=child.stride if hasattr(child, 'stride') else 1,
            #     downsample=child.downsample,
            #     groups=child.groups,
            #     base_width=child.base_width,
            #     dilation=child.dilation,
            #     se_reduction=se_reduction
            # )

            se_block = SEBottleneck(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                groups=groups,
                base_width=base_width,
                dilation=dilation,
                norm_layer=norm_layer,
                se_reduction=se_reduction
            )

            se_block.conv1.load_state_dict(child.conv1.state_dict())
            se_block.bn1.load_state_dict(child.bn1.state_dict())
            se_block.conv2.load_state_dict(child.conv2.state_dict())
            se_block.bn2.load_state_dict(child.bn2.state_dict())
            se_block.conv3.load_state_dict(child.conv3.state_dict())
            se_block.bn3.load_state_dict(child.bn3.state_dict())
            if child.downsample is not None:
                se_block.downsample.load_state_dict(child.downsample.state_dict())

            setattr(module, name, se_block)
        else:
            _replace_bottleneck_with_se(child, se_reduction=se_reduction)
    
def build_se_resnet50(
    num_classes=5,
    freeze_backbone=False,
    unfreeze_last_layers=1,
    se_reduction=16
):
    model = models.resnet50(weights="IMAGENET1K_V1")

    _replace_bottleneck_with_se(model, se_reduction=se_reduction)

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

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def count_trainable_params(model):
    """Đếm số tham số train được."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

if __name__ == "__main__":
    print("=" * 60)
    print("SE-ResNet50: Building & Testing")
    print("=" * 60)
 
    # Build SE-ResNet50
    model = build_se_resnet50(
        num_classes=5,
        freeze_backbone=True,
        unfreeze_last_layers=1,
        se_reduction=16,
    )
 
    trainable, total = count_trainable_params(model)
    print(f"\nSE-ResNet50")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"  Frozen params:    {total - trainable:,} ({100*(total-trainable)/total:.1f}%)")
 
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"\nForward pass test:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  ✓ SE-ResNet50 works correctly!")