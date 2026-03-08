import torch
from torch import nn
from layers import *

class Backbone(nn.Module):
    def __init__(
        self, 
        depth: float = 1.0,
        width: float = 1.0,
        base_channels: list = [64,128,256,512,1024],
        ):
        super().__init__()
        in_channels = [min(int(ch * width), 512) for ch in base_channels] #限制最高通道数

        self.m1 = nn.Sequential(
            Conv(3, int(in_channels[0]), 3, 2),
            Conv(in_channels[0], in_channels[1], 3, 2),
            C3k2(in_channels[1], in_channels[2], int(2 * depth), 0.25, False),
            Conv(in_channels[2], in_channels[2], 3, 2),
            C3k2(in_channels[2], in_channels[3], int(2 * depth), 0.25, False),
        )
        self.m2 = nn.Sequential(
            Conv(in_channels[3], in_channels[3], 3, 2),
            A2C2f(in_channels[3], int(4 * depth), 4)
        )
        self.m3 = nn.Sequential(
            Conv(in_channels[3], in_channels[4], 3, 2),
            A2C2f(in_channels[4], int(4 * depth), 1) #注意此处area参数为1
        )
    
    def forward(self, x: torch.Tensor):
        P3 = self.m1(x) #80*80*512
        P4 = self.m2(P3) #40*40*512
        P5 = self.m3(P4) #20*20*1024
        return (P3,P4,P5)
    
class Head(nn.Module):
    def __init__(
        self, 
        depth: float = 1.0,
        width: float = 1.0,
        channels: list = [256,512,1024],
        ):
        super().__init__()
        in_channels = [min(int(ch * width), 512) for ch in channels] #限制最高通道数

        self.upsample = nn.Upsample(size = None, scale_factor = 2, mode = "nearest")
        self.A12 = A2_for_head((in_channels[1] + in_channels[2]), in_channels[1],
                               int(2 * depth), 1)
        self.A11 = A2_for_head((in_channels[1] + in_channels[1]), in_channels[0],
                               int(2 * depth), 1)        
        self.cv21 = Conv(in_channels[0], in_channels[0], 3, 2)
        self.A22 = A2_for_head((in_channels[0] + in_channels[1]), in_channels[1],
                               int(2 * depth), 1) 
        self.cv23 = Conv(in_channels[1], in_channels[1], 3, 2)
        self.C24 = C3k2((in_channels[1] + in_channels[2]), in_channels[2], int(2 * depth), 0.5, True)

    def forward(self, inputs: tuple):
        t1, t2, t3 = inputs

        u13 = self.upsample(t3)
        t12 = self.A12(torch.cat([t2, u13], 1))

        u12 = self.upsample(t12)
        out1 = self.A11(torch.cat([t1, u12], 1)) #80*80*256

        c21 = self.cv21(out1)
        out2 = self.A22(torch.cat([t12, c21], 1)) #40*40*512

        c23 = self.cv23(out2)
        out3 = self.C24(torch.cat([t3, c23], 1)) #20*20*1024

        return (out1,out2,out3)
    



