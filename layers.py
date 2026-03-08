import torch
from torch import nn

class Conv(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        act: str = "silu"
        ):
        super().__init__()
        # 定义卷积、BN、激活函数
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # 选择激活函数（默认使用 SiLU 激活函数）
        if act == "silu":
            self.act = nn.SiLU()
        elif act == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
#无激活函数的卷积层
class ConvBN(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        ):
        super().__init__()

        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))
    
#深度可分离卷积
class DWConv(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        ksize: int, 
        stride: int = 1, 
        act: str = "silu"
        ):
        super().__init__()
        # 深度卷积 (Depthwise): groups=in_channels
        self.dconv = Conv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)
        # 逐点卷积 (Pointwise): 1x1 卷积
        self.pconv = Conv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        return self.pconv(self.dconv(x))

#area attention    
class AAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        area: int = 1
        ):
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.all_head_dim = all_head_dim = head_dim * self.num_heads #使其能被num_heads整除

        self.qkv = ConvBN(dim, all_head_dim * 3, 1)
        self.proj = ConvBN(all_head_dim, dim, 1)
        self.pe = ConvBN(all_head_dim, all_head_dim, 7, groups=dim)

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        N = H * W #此处要求N为area(可能为4等）的倍数
        
        qkv=self.qkv(x).flatten(2).transpose(1,2) #(B, C, H, W)->(B, N, C2=3*all_head_dim)
        if self.area > 1:
            if not torch.jit.is_tracing():
                assert N % self.area == 0, f"Internal Error: N({N}) is not divisible by area({self.area})"
            qkv =qkv.reshape(B * self.area, N // self.area, self.all_head_dim * 3) #(B, N, C2)->(B2, N2, C2)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3).permute(0,2,3,1)
            .split([self.head_dim, self.head_dim, self.head_dim],dim=2)
        ) #q,k,v.shape:(B2, num_heads, head_dim, N2)

        attn = (q.transpose(-2,-1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1) #attn.shape:(B2, num_heads, N2, N2)
        x = v @ attn.transpose(-2,-1) # -> (B2, num_heads, head_dim, N2)

        x = x.permute(0, 3, 1, 2)  
        v = v.permute(0, 3, 1, 2)
        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, self.all_head_dim)
            v = v.reshape(B // self.area, N * self.area, self.all_head_dim)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
        #x,v.shape:(B, all_head_dim, H, W)
        x = x + self.pe(v)
        return self.proj(x)
    
class ABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        area: int = 1
        ):
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), ConvBN(mlp_hidden_dim, dim, 1))

    def forward(self, x: torch.Tensor):
        x = x + self.attn(x)
        return x + self.mlp(x)

"""官方代码在A2C2f中通过shortcut参数区分，操作不直观，
   故此处将shortcut=True的情况写为A2C2f类，注意到其输出通道=输入通道的特点，故将两个通道参数写为同一个；
   由于shortcut=False的情况仅在Head部分出现，故将其写为另一类A2_for_head"""
class A2C2f(nn.Module):
    def __init__(
        self, 
        channels: int,
        n: int,
        area: int = 1,
        mlp_ratio: float = 2.0, #此处传参决定ABlock中mlp通道增加倍数
        e: float = 0.5
        ):
        super().__init__()
        hidden_channels = int(channels * e) #要求结果为32的倍数

        self.cv1 = Conv(channels, hidden_channels, 1)
        self.cv2 = Conv((1 + n) * hidden_channels, channels, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(channels), requires_grad=True) #加权残差连接，确保训练前期稳定
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(hidden_channels, hidden_channels // 32, mlp_ratio, area) for _ in range(2)))
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y,1))
        return x + self.gamma.view(-1,self.gamma.shape[0],1,1) * y
   
class A2_for_head(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        n: int,
        area: int = 1,
        mlp_ratio: float = 2.0, #此处传参决定ABlock中mlp通道增加倍数
        e: float = 0.5
        ):
        super().__init__()
        hidden_channels = int(out_channels * e) #要求结果为32的倍数

        self.cv1 = Conv(in_channels, hidden_channels, 1)
        self.cv2 = Conv((1 + n) * hidden_channels, out_channels, 1)    

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(hidden_channels, hidden_channels // 32, mlp_ratio, area) for _ in range(2)))
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor):
        y = [self.cv1(x)]
        y.extend(layer(y[-1]) for layer in self.m)
        y = self.cv2(torch.cat(y, 1))
        return y
    

class C3k(nn.Module):
    def __init__(
        self,
        channels: int,       
        shortcut: bool = True,
        groups: int = 1
        ):
        super().__init__()
        self.shortcut = shortcut

        self.cv1 = Conv(channels, channels, 3)
        self.cv2 = Conv(channels, channels, 3, groups=groups)

    def forward(self, x: torch.Tensor):
        if self.shortcut:
            return x + self.cv2(self.cv1(x))
        return self.cv2(self.cv1(x))

class C3k2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        e: float = 0.5,
        shortcut: bool = True,
        groups: int = 1 #此处传参决定C3k中第二次卷积的方式
        ):
        super().__init__()

        hidden_channels = int(out_channels * e)

        self.cv1 = Conv(in_channels, hidden_channels * 2, 1)
        self.cv2 = Conv(hidden_channels * (2 + n), out_channels, 1)
        self.m = nn.ModuleList(
            C3k(hidden_channels, shortcut, groups)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor):
        y = list(self.cv1(x).chunk(2,1))
        y.extend(layer(y[-1]) for layer in self.m)
        return self.cv2(torch.cat(y, 1))

