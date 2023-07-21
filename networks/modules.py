import torch.nn as nn
import torch


class Unsqueeze(nn.Module):

    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def forward(self, input):
        factor = self.factor
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        B, C, H, W = input.shape
        factor2 = factor**2
        assert C % (factor2) == 0, "{}".format(C)
        x = input.view(B, C // factor2, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x


class Squeeze(nn.Module):

    def __init__(self, factor):
        super.__init__()
        self.factor = factor

    def forward(self, input):
        factor = self.factor
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        B, C, H, W = input.shape
        factor2 = factor**2
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * factor2, H // factor, W // factor)
        return x


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
    if kernel_size == 3:
        padding = 1
    elif kernel_size == 5:
        padding = 2
    elif kernel_size == 7:
        padding = 3
    elif kernel_size == 1:
        padding = 0
    else:
        raise NotImplementedError('not implemented...')

    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     bias=True)


class Bottleneck(nn.Module):

    def __init__(self, c_in) -> None:
        super().__init__()
        self.conv1 = conv(c_in, c_in)
        self.conv2 = conv(c_in, c_in)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.relu(x + y)


class ZeroConv2d(nn.Module):
    '''The 3x3 convolution in which weight and bias are initialized with zero.
    The output is then scaled with a positive learnable param.
    '''

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = self.conv(input)
        out = out * torch.exp(self.scale * 3)
        return out


class FlowHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
