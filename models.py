import os
import numpy as np
import glob
import torch
import random
import torchvision
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
######### Resnet ##############
###############################


class Model_OneLeg(nn.Module):
    """
    Same model with the first layer modified.
    """

    def __init__(self, model, n_class):
        super().__init__()
        self._conv_stem = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.rest = nn.Sequential(*list(model.children())[1:-1])
        self.linear = nn.Linear(in_features=2048, out_features=n_class, bias=True)
        
    def forward(self, input):
        out = self._conv_stem(input)
        out = self.rest(out)
        out = self.linear(out.squeeze())
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
def Get_Pytorch_Model(model='resnet152', pretrained=True, n_class=5):
    if model == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    if model == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    if model == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
    print('number of model parameters: {}'.format(count_parameters(model)))
    oneleg = Model_OneLeg(model, n_class)
    return oneleg
    
    
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, last_size=4928, num_classes=5):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(24)
        self.fc = nn.Linear(last_size, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out