import torchvision
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F


class adjustedResNet50(nn.Module):

    def __init__(self, num_classes=8):

        super(adjustedResNet50, self).__init__()

        self.model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # remove the final connected layer by putting a placeholder
        self.model.fc = nn.Identity()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048,1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):

        x = self.model(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x