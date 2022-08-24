
import torch
import torch.nn as nn
from utils.config import ConfigDict

class DownStream(nn.Module):
    def __init__(self, backbone: nn.Module, args: ConfigDict):
        
        super(DownStream, self).__init__()
        expansion={'18': 1, '50': 4}
        
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion[args['depth']], args['num_classes'])
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x