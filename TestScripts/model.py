import torch.nn as nn
import torch
from torchvision.models import vgg19
import torchvision 

'''
EnhanceNet Implementation in PyTorch by Erik Quintanilla 
Single Image Super Resolution 
https://arxiv.org/abs/1612.07919/
This program assumes GPU.
'''

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        ip_ = x
        return torch.add(self.conv_block(x), ip_)
        
        
class Model(nn.Module):
    def __init__(self, in_channels=8, out_channels=2, residual_blocks=64):
        super(Model, self).__init__()
        self.merge = torch.cat
        self.add = torch.add
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1), 
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
                nn.ReLU())
        #Residual blocks
        residuals = []
        for _ in range(residual_blocks):
            residuals.append(ResidualBlock(64))
        self.residuals = nn.Sequential(*residuals)
        
        #nearest neighbor upsample 
        self.seq = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residuals(out)
        out = self.conv3(out)
        out = self.conv4(out) 

        return out