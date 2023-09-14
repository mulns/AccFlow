import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from .modules import ZeroConv2d
from .raft.extractor import BasicEncoder
from .utils import backwarp

autocast = torch.cuda.amp.autocast


class FlowDecoder(nn.Module):
    def __init__(self, cin=128):
        super().__init__()
        self.flow = nn.Sequential(
            nn.Conv2d(cin, cin * 2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(cin * 2, 2, 3, 1, 1),
        )
        self.mask = nn.Sequential(
            nn.Conv2d(cin, cin * 2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(cin * 2, 64 * 9, 1, 1, 0),
        )

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, x):
        flow_small = self.flow(x)
        mask = self.mask(x)
        flow = self.upsample_flow(flow_small, mask)

        return flow_small, flow


class FlowEncoder(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.conv1 = nn.Conv2d(2, c, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(c, c * 2, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(c * 2, c, 1, stride=1, padding=0)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x)))))
        if is_list:
            x = torch.split(x, batch_dim, dim=0)
        return x


class AccPlus(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c * 2 + 1, c * 2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(c * 2, c, 3, 1, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(c * 2, c, 3, 1, 1),
            nn.ReLU(True),
            ZeroConv2d(c, 3**3),
        )
        self.dconv = DeformConv2d(c, c, 3, 1, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(c * 2 + 1, c * 2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(c * 2, c, 3, 1, 1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(c * 4, c * 2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(c * 2, c, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(c, c, 1, 1, 0),
        )

    def forward(self, df, f, o, c):
        x = torch.cat([df, f, o], dim=1)
        x = self.conv1(x)
        x = torch.cat([x, c], dim=1)
        x = self.conv2(x)
        off, m = torch.split(x, [18, 9], dim=1)
        m = torch.sigmoid(m)
        f_ = self.dconv(f, off, m)
        x = torch.cat([f_, df, o], dim=1)
        x = self.conv3(x)
        x = torch.cat([x, c, f_, df], dim=1)
        x = self.conv4(x)
        return x


class Blending(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(c, c * 2, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(c * 2, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, f1, f2, emap):
        m = self.mask(emap)
        return f1 * m + (1 - m) * f2


def getOcc(F12, I1, I2, binary=True):
    I1_ = backwarp(I2, F12)
    e = torch.abs(I1 - I1_)
    if binary:
        e = torch.mean(e, dim=1, keepdim=True)
        ones = torch.ones_like(e, device=e.device)
        zeros = torch.zeros_like(e, device=e.device)
        return torch.where(e <= 1.0, ones, zeros)
    return e


def downflow8(flow, mode="bilinear"):
    h, w = flow.shape[-2:]
    assert h % 8 == 0 and w % 8 == 0
    flow = F.interpolate(flow, size=(h // 8, w // 8), mode=mode, align_corners=True)
    return flow / 8


class AccFlow(nn.Module):
    def __init__(self, ofe: nn.Module):
        super().__init__()
        self.ofe: nn.Module = ofe
        self.hidden_channel = 128
        self.flow_encoder = FlowEncoder(self.hidden_channel)
        self.flow_decoder = FlowDecoder(self.hidden_channel)
        self.context = BasicEncoder(3, output_dim=self.hidden_channel, norm_fn="none")
        self.accplus = AccPlus(self.hidden_channel)
        self.blending = Blending(self.hidden_channel)
        self.mixed_precision = True

    def forward(self, images, test_mode=False):  # FIXME
        """
        Input: [I1, I2, ..., In]
        Output: If test_mode=False:
                  [F31, F41, ..., Fn1]
                If test_mode=True:
                  [F21, F31, ..., Fn1]
        """
        flow = None
        outs = []
        for i in range(2, len(images)):
            I1 = images[i]
            I2 = images[i - 1]
            In = images[0]
            if flow is not None:
                flow = flow.detach()
            flow, flow_up = self.iter(I1, I2, In, flow)
            outs.append(flow_up)
        return outs

    def iter(self, I1, I2, In, F2n):
        """
        input: I1, I2, IN; F2N (1/8 size)
        output: F1N_small (1/8 size), F1N
        """
        with torch.no_grad():
            if F2n is None:
                flows = self.ofe(torch.cat([I1, I1, I2]), torch.cat([I2, In, In]))
                flows = downflow8(flows)
                dflow, flow_ini, F2n = flows.chunk(3)
            else:
                flows = self.ofe(torch.cat([I1, I1]), torch.cat([I2, In]))
                flows = downflow8(flows)
                dflow, flow_ini = flows.chunk(2)
        with autocast(enabled=self.mixed_precision):
            f_ini, df, f = self.flow_encoder([flow_ini, dflow, F2n])
            c1, c2, cn = self.context([I1, I2, In])
            o = getOcc(dflow, c1, c2)
            o = o.detach()
            f_acc = self.accplus(df, f, o, c1)
            emap = getOcc(flow_ini, c1, cn, binary=False)
            emap = emap.detach()
            f_fuse = self.blending(f_ini, f_acc, emap)
            out_small, out = self.flow_decoder(f_fuse)
        return out_small.float(), out.float()
