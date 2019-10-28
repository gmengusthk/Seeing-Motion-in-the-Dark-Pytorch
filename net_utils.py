import torch
import torch.nn as nn

class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x

class SpaceToDepth(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x

def conv2d(in_channels,out_channels,kernel_size=3,stride=1,activation=True,use_bn=False):
    if use_bn:
        conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=int((kernel_size-1)/2),bias=False,stride=stride)
        bn=nn.BatchNorm2d(out_channels)
        if activation:
            lrelu=nn.LeakyReLU(negative_slope=0.2)
            return nn.Sequential(*[conv,bn,lrelu])
        else:
            return nn.Sequential(*[conv,bn])
        
    else:
        conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=int((kernel_size-1)/2),bias=True,stride=stride)
        if activation:
            lrelu=nn.LeakyReLU(negative_slope=0.2)
            return nn.Sequential(*[conv,lrelu])

        else:
            return nn.Sequential(*[conv])


def deconv_2d(in_channels,out_channels,use_bn=False):
    if use_bn:
        deconv=nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        bn=nn.BatchNorm2d(out_channels)
        return nn.Sequential(*[deconv,bn])
    else:
        deconv=nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True)
        return nn.Sequential(*[deconv])
