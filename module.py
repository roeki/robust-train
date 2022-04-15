import torch
import torch.nn as nn
import math
import copy
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter


# 权重定义与正常方法一样，前向传播时分别计算即可
class RobustLinear(nn.Module):
    # 输入，输出，是否有偏置，权重是否非负
    def __init__(self, in_features, out_features, bias=True, non_negative = True):
        super(RobustLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if non_negative:
            # rand 生成均匀分布的伪随机数。分布在（0~1）之间
            self.weight = Parameter(torch.rand(out_features, in_features) * 1/math.sqrt(in_features))
        else:
            # randn 生成标准正态分布的伪随机数（均值为0，方差为1）
            self.weight = Parameter(torch.randn(out_features, in_features) * 1/math.sqrt(in_features))

        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        self.non_negative = non_negative

    def forward(self, input):
        # 输入分别为上下界，分别取出
        input_p = input[:input.shape[0]//2]
        input_n = input[input.shape[0]//2:]
        # 如果权重非负，则单调 那么计算上下界只需要两个矩阵乘法
        if self.non_negative:
            out_p = F.linear(input_p, F.relu(self.weight), self.bias)
            out_n = F.linear(input_n, F.relu(self.weight), self.bias)
            # torch.cat(inputs, dimension=0) → Tensor //dimension (int, optional) – 沿着此维连接张量序列。
            return torch.cat([out_p, out_n], 0)
        # 如果权重不是非负，则用文章提到的方法。用更快的方法计算一个相对宽一些的界限
        u = (input_p + input_n)/2
        r = (input_p - input_n)/2
        out_u = F.linear(u, self.weight, self.bias)
        out_r = F.linear(r, torch.abs(self.weight), None)
        return torch.cat([out_u + out_r, out_u - out_r], 0)


class RobustConv2d(nn.Module):
    # cove1d：用于文本数据，只对宽度进行卷积，对高度不进行卷积
    # cove2d：用于图像数据，对宽度和高度都进行卷积
    # 卷积神将网络的计算公式为：
    # N=(W-F+2P)/S+1
    # 其中 输入的通道数:in_channels  输出的通道数:out_channels
    # N：输出大小
    # W：输入大小
    # F：卷积核大小 kernel*kenel
    # P：填充值的大小 padding
    # S：步长大小 stride
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, non_negative = True):
        super(RobustConv2d, self).__init__()
        # 权重的初始化，是否非负
        if non_negative:
            self.weight = Parameter(torch.rand(out_channels, in_channels//groups, kernel_size, kernel_size) * 1/math.sqrt(kernel_size * kernel_size * in_channels//groups))
        else:
            self.weight = Parameter(torch.randn(out_channels, in_channels//groups, kernel_size, kernel_size) * 1/math.sqrt(kernel_size * kernel_size * in_channels//groups))
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.non_negative = non_negative

    def forward(self, input):
        input_p = input[:input.shape[0]//2]
        input_n = input[input.shape[0]//2:]
        if self.non_negative:
            out_p = F.conv2d(input_p, F.relu(self.weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
            out_n = F.conv2d(input_n, F.relu(self.weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
            return torch.cat([out_p, out_n],0)

        u = (input_p + input_n)/2
        r = (input_p - input_n)/2
        out_u = F.conv2d(u, self.weight,self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        out_r = F.conv2d(r, torch.abs(self.weight), None, self.stride,
                        self.padding, self.dilation, self.groups)
        return torch.cat([out_u + out_r, out_u - out_r], 0)


class ImageNorm(nn.Module):
    def __init__(self, mean, std):
        super(ImageNorm, self).__init__()
        self.mean = torch.from_numpy(np.array(mean)).view(1,3,1,1).cuda().float()
        self.std = torch.from_numpy(np.array(std)).view(1,3,1,1).cuda().float()

    def forward(self, input):
        input = torch.clamp(input, 0, 1)
        return (input - self.mean)/self.std
