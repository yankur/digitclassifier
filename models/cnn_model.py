import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..interface import DigitClassificationInterface


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) # 1x28x28 -> 32x28x28
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # 32x14x14 -> 64x14x14
        self.fc3 = nn.Linear(in_features=64*7*7, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*7*7) # flatten
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class CNNModel(DigitClassificationInterface):
    def __init__(self):
        self.model = CNN()
        self.model.eval() # random weights init

    def predict(self, image: torch.tensor) -> int:
        with torch.no_grad():
            output = self.model(image)
            pred = output.argmax(dim=1, keepdim=True).item()
        return pred




