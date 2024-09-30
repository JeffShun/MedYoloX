import torch
import torch.nn as nn
from typing import List

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 8, c2, k, s, p, g, act)

    def forward(self, x):
        # Split, concatenate and apply convolution
        return self.conv(
            torch.cat([
                x[..., ::2, ::2, ::2], 
                x[..., 1::2, ::2, ::2], 
                x[..., ::2, 1::2, ::2], 
                x[..., ::2, ::2, 1::2], 
                x[..., 1::2, 1::2, ::2], 
                x[..., 1::2, ::2, 1::2], 
                x[..., ::2, 1::2, 1::2], 
                x[..., 1::2, 1::2, 1::2]
            ], 1)
        )
    
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv3d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm3d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat([
                self.m(self.cv1(x)), 
                self.cv2(x)
                ], 1)
            )

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool3d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
        
class CSPDarknet3D(nn.Module):
    def __init__(self, in_channel=1, base_channel=32, base_depth=3):
        super().__init__()
        self.stem = Focus(in_channel, base_channel)
        self.dark2 = nn.Sequential(
            Conv(base_channel, base_channel, 3, 2),
            C3(base_channel, base_channel, base_depth),
        )
        self.dark3 = nn.Sequential(
            Conv(base_channel, base_channel * 2, 3, 2),
            C3(base_channel * 2, base_channel * 2, base_depth * 3),
        )
        self.dark4 = nn.Sequential(
            Conv(base_channel * 2, base_channel * 4, 3, 2),
            C3(base_channel * 4, base_channel * 4, base_depth * 3),
        )
        self.dark5 = nn.Sequential(
            Conv(base_channel * 4, base_channel * 8, 3, 2),
            SPP(base_channel * 8, base_channel * 8),
            C3(base_channel * 8, base_channel * 8, base_depth, shortcut=False),
        )
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = x
        x = self.dark4(x)
        feat2 = x
        x = self.dark5(x)
        feat3 = x
        return [feat1, feat2, feat3]
    