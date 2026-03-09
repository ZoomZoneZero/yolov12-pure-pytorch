import torch
from torch import nn
from layers import *

class DFL(nn.Module):
    def __init__(
        self,
        reg_max: int
        ):
        super().__init__()
        self.reg_max = reg_max
        # 直接加载 1-regmax 的序列作为卷积权重
        self.cv = nn.Conv2d(reg_max, 1, 1, bias = False).requires_grad_(False)
        r = torch.arange(reg_max, dtype = torch.float)
        self.cv.weight.data.copy_(r.view(1, reg_max, 1, 1))

    def forward(self, x: torch.Tensor):
        B, _, N = x.shape
        return self.cv(x.contiguous().view(B * 4, self.reg_max, 1, N).softmax(1)).view(B, 4, N)

class Dist2bbox(nn.Module):
    def __init__(
        self, 
        stride: int,
        xywh: bool = False  # xywh: 返回格式开关。True 返回 [cx, cy, w, h]，False 返回 [x1, y1, x2, y2]
        ):
        super().__init__()
        self.stride = stride
        self.xywh = xywh

    @staticmethod
    def get_anchors(x: torch.Tensor, size: tuple):
        device = x.device
        h, w = size
        
        shift_x = torch.arange(w, device=device, dtype=torch.float32) + 0.5
        shift_y = torch.arange(h, device=device, dtype=torch.float32) + 0.5

        grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        
        anchor_points = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2).transpose(0, 1).unsqueeze(0)
        return anchor_points


    def forward(self, x: torch.Tensor, size: tuple) -> torch.Tensor:
        anchor_points = self.get_anchors(x, size)
        anchor_points = anchor_points * self.stride

        lt, rb = torch.split(x, 2, dim=1)
        x1y1 = anchor_points - (lt * self.stride)
        x2y2 = anchor_points + (rb * self.stride)

        if self.xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh   = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim=1) # [B, 4, N]
        
        return torch.cat((x1y1, x2y2), dim=1) # [B, 4, N]

class Detect(nn.Module):
    def __init__(
        self,
        num_cls: int = 80, 
        reg_max: int = 16, 
        width: float = 1.0,
        channels: list = [256, 512, 1024],
        strides: tuple = (8, 16, 32),
        xywh: bool = False  # xywh: 返回格式开关。True 返回 [cx, cy, w, h]，False 返回 [x1, y1, x2, y2]
        ):
        super().__init__()
        in_channels = [min(int(ch * width), 512) for ch in channels]

        self.num_cls = num_cls
        self.reg_max = reg_max # DFL channels (in_channels[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x),后续代码会传入
        ch_reg = max((16, in_channels[0] // 4, self.reg_max * 4)) # 利用 reg_max 约束了下限，确保 DFL 顺利进行
        ch_cls = max(in_channels[0], min(self.num_cls, 100)) # 在类别极多时压缩冗余计算
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(ch, ch_reg, 3),
                Conv(ch_reg, ch_reg, 3),
                nn.Conv2d(ch_reg, self.reg_max * 4, 1)
            ) for ch in in_channels
        )
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                DWConv(ch, ch, 3), Conv(ch, ch_cls, 1),        
                DWConv(ch_cls, ch_cls, 3), Conv(ch_cls, ch_cls, 1),
                nn.Conv2d(ch_cls, self.num_cls, 1)
            ) for ch in in_channels
        )
        self.dfl = DFL(self.reg_max)
        self.box = nn.ModuleList(
            Dist2bbox(stride, xywh) for stride in strides
        )

    def forward(self, inputs: tuple):
        y = []
        for i, x in enumerate(inputs):
            box_out = self.box[i](self.dfl(self.cv1[i](x).flatten(2)), x.shape[-2:])
            cls_out = self.cv2[i](x).flatten(2).sigmoid()
            y.append(torch.cat([box_out, cls_out], dim=1))
        return torch.cat(y, dim=2).permute(0, 2, 1)

class Detect_train(nn.Module):
    def __init__(
        self,
        num_cls: int = 80, 
        reg_max: int = 16, 
        width: float = 1.0,
        channels: list = [256, 512, 1024],
        strides: tuple = (8, 16, 32),
        xywh: bool = False 
        ):
        super().__init__()
        in_channels = [min(int(ch * width), 512) for ch in channels]

        self.num_cls = num_cls
        self.reg_max = reg_max 
        
        ch_reg = max((16, in_channels[0] // 4, self.reg_max * 4)) 
        ch_cls = max(in_channels[0], min(self.num_cls, 100)) 

        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(ch, ch_reg, 3),
                Conv(ch_reg, ch_reg, 3),
                nn.Conv2d(ch_reg, self.reg_max * 4, 1)
            ) for ch in in_channels
        )
        
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                DWConv(ch, ch, 3), Conv(ch, ch_cls, 1),        
                DWConv(ch_cls, ch_cls, 3), Conv(ch_cls, ch_cls, 1),
                nn.Conv2d(ch_cls, self.num_cls, 1)
            ) for ch in in_channels
        )
        
        self.dfl = DFL(self.reg_max)
        self.box = nn.ModuleList(
            Dist2bbox(stride, xywh) for stride in strides
        )
        
        self._initialize_biases(strides)

    def _initialize_biases(self, strides, cf=None):
        """
        初始化分类分支的偏置，解决 Loss 初始值过高的问题。
        """
        import math
        for m, stride in zip(self.cv2, strides): 
            last_conv = None
            if isinstance(m, nn.Sequential):
                for layer in reversed(m):
                    if isinstance(layer, nn.Conv2d):
                        last_conv = layer
                        break
            else:
                last_conv = m
            
            if last_conv is None:
                continue

            # 先验概率
            if cf is None:
                prior = 8 / self.num_cls / (640 / stride) ** 2
            else:
                prior = cf

            # 计算 Bias 值 (防止 log(0))
            bias_value = math.log(prior / (1 - prior + 1e-6))
            
            with torch.no_grad():
                last_conv.bias.fill_(bias_value)
                
    def forward(self, inputs: tuple):
        box_outs = []
        cls_outs = []
        reg_outs = []
        all_anchors = []
        all_strides = []

        for i, x in enumerate(inputs):
            h, w = x.shape[-2:]
            stride = self.box[i].stride 

            raw_reg = self.cv1[i](x).flatten(2)   
            cls_out = self.cv2[i](x).flatten(2)   
            
            dist = self.dfl(raw_reg)
            box_out = self.box[i](dist, (h, w)) 
            
            # 锚点生成 (用于 Loss)
            anchor_grid = Dist2bbox.get_anchors(x, (h, w))
            all_anchors.append(anchor_grid * stride) 
            
            s_tensor = torch.full((1, 1, h * w), stride, device=x.device, dtype=x.dtype)
            all_strides.append(s_tensor)

            box_outs.append(box_out)
            cls_outs.append(cls_out)
            reg_outs.append(raw_reg)

        # 拼接所有层级的结果
        # box: [B, 4, 8400]
        # cls: [B, nc, 8400]
        # reg: [B, 64, 8400]
        # anchors: [1, 2, 8400]
        # strides: [1, 1, 8400]
        return (torch.cat(box_outs, dim=2), 
                torch.cat(cls_outs, dim=2), 
                torch.cat(reg_outs, dim=2), 
                torch.cat(all_anchors, dim=2), 
                torch.cat(all_strides, dim=2))