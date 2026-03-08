import torch
from torch import nn
from layers import *
from framework import *
from detect import *

class Yolo12(nn.Module):
    def __init__(
        self, 
        scales: str = "s",
        num_cls: int = 80,
        xywh: bool = False        
        ):
        super().__init__()

        depth_dict = {'n': 0.50, 's' : 0.50, 'm' : 0.50, 'l' : 1.00, 'x' : 1.00,}
        width_dict = {'n': 0.25, 's' : 0.50, 'm' : 1.00, 'l' : 1.00, 'x' : 1.50,}

        base_channels = [64,128,256,512,1024]
        channels = base_channels[-3:]
        reg_max = int(channels[0] * width_dict[scales]) // 16
        strides = (8, 16, 32)

        self.backbone = Backbone(depth=depth_dict[scales], width=width_dict[scales], base_channels=base_channels)
        self.head     = Head(depth=depth_dict[scales], width=width_dict[scales], channels=channels)
        self.detect   = Detect(num_cls=num_cls, reg_max=reg_max, width=width_dict[scales], channels=channels, strides=strides, xywh=xywh)

    def forward(self, x: torch.Tensor):
        output = self.detect(self.head(self.backbone(x)))
        return output

class Yolo12_train(nn.Module):
    def __init__(
        self, 
        scales: str = "s",
        num_cls: int = 80,
        xywh: bool = False        
        ):
        super().__init__()

        depth_dict = {'n': 0.50, 's' : 0.50, 'm' : 0.50, 'l' : 1.00, 'x' : 1.00,}
        width_dict = {'n': 0.25, 's' : 0.50, 'm' : 1.00, 'l' : 1.00, 'x' : 1.50,}

        base_channels = [64,128,256,512,1024]
        channels = base_channels[-3:]
        reg_max = int(channels[0] * width_dict[scales]) // 16
        strides = (8, 16, 32)

        self.backbone = Backbone(depth=depth_dict[scales], width=width_dict[scales], base_channels=base_channels)
        self.head     = Head(depth=depth_dict[scales], width=width_dict[scales], channels=channels)
        self.detect   = Detect_train(num_cls=num_cls, reg_max=reg_max, width=width_dict[scales], channels=channels, strides=strides, xywh=xywh)

    def forward(self, x: torch.Tensor):
        output = self.detect(self.head(self.backbone(x)))
        return output
