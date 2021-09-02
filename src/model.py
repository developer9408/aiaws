import torchvision
import torch
import torch.nn as nn


class NutriNet(nn.Module):
    def __init__(self, pretrained=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        # for param in self.resnet.parameters():
            # param.requires_grad = False

        self.resnet.fc = nn.Linear(512, 6)

    def forward(self, x):
        return self.resnet(x)


