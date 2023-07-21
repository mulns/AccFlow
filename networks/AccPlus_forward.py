import argparse
import torch
from torch import nn as nn
import logging
from networks.raft.raft import RAFT
from torchvision.ops import DeformConv2d
from .raft.extractor import BasicEncoder
from .modules import conv, ZeroConv2d, FlowHead
from networks.raft.softsplat import Softsplat

logger = logging.getLogger('base')  # base logger


class FlowAcc(nn.Module):
    '''Inputs: (f, df): motion features in (N,C,H,W)
    '''

    def __init__(self, in_c=128) -> None:
        super().__init__()
        self.head = nn.Sequential(conv(in_c * 2, 256), nn.ReLU(),
                                  conv(256, 128))
        self.offset1 = nn.Sequential(conv(in_c * 2, 256), nn.ReLU(),
                                     conv(256, 128), nn.ReLU(),
                                     ZeroConv2d(128, 3**3))
        self.dconv1 = DeformConv2d(in_c, in_c, 3, 1, 1)
        self.merge1 = conv(in_c * 2, in_c)

        self.tail = nn.Sequential(conv(in_c * 2, 256), nn.ReLU(),
                                  conv(256, 256), nn.ReLU(), conv(256, 128))

        self.to_mask = nn.Sequential(conv(256, 512), nn.ReLU(), conv(512, 1),
                                     nn.Sigmoid())
        self.merge2 = nn.Sequential(conv(in_c * 2 + 1, 512), nn.ReLU(),
                                    conv(512, in_c))

    def forward(self, f, df, c, f_ofe, emap):
        x = torch.cat([f, df], dim=1)
        x = self.head(x)
        x = torch.cat([x, c], dim=1)
        off, m = torch.split(self.offset1(x), [18, 9], dim=1)
        m = torch.sigmoid(m)
        x = self.dconv1(df, off, m)
        x = torch.cat([x, f], dim=1)
        x = self.merge1(x)
        x = torch.cat([x, c], dim=1)
        x = self.tail(x)

        m = self.to_mask(emap)
        x = self.merge2(torch.cat([x, f_ofe, m], dim=1))
        return x


class AccPlus(nn.Module):

    def __init__(self, ofe='raft-things') -> None:
        super().__init__()
        hidden_channel = 128
        self.flowacc = FlowAcc(hidden_channel)
        self.flow_head = FlowHead(hidden_channel, 256)
        self.mask_head = nn.Sequential(conv(hidden_channel, 256), nn.ReLU(),
                                       conv(256, 64 * 9, 1))

        if 'raft' in ofe.lower():
            raft_args = argparse.Namespace(small=False, mixed_precision=True)
            raft = RAFT(raft_args)
            if 'things' in ofe.lower():
                ckpt = torch.load('./pretrained/raft-things.pth')
            else:
                ckpt = torch.load('./pretrained/raft-kubric.pth')
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            raft.load_state_dict(ckpt, strict=True)
            self.ofe = raft
        elif 'gma' in ofe.lower():
            from networks.gma.gma import RAFTGMA
            gma_args = argparse.Namespace(num_heads=1,
                                          mixed_precision=True,
                                          position_only=False,
                                          position_and_content=False)
            gma = RAFTGMA(gma_args)
            if 'things' in ofe.lower():
                ckpt = torch.load('./pretrained/gma-things.pth')
            else:
                ckpt = torch.load('./pretrained/gma-kubric.pth')
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            gma.load_state_dict(ckpt, strict=True)
            self.ofe = gma

        params = sum(p.numel() for p in self.ofe.parameters()
                     if p.requires_grad)
        logger.info('Parmameter of Optical Flow Estimator: %d' % params)
        for p in self.ofe.parameters():
            p.requires_grad = False

        self.fnet = nn.Sequential(nn.Conv2d(2, 128, 7, stride=1, padding=3),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(256, 128, 1, stride=1, padding=0))
        self.cnet = BasicEncoder(3, output_dim=128, norm_fn='none', dropout=0)

        self.mixed_precision = True

    def forward(self, images, iters=12, test_mode=False):
        images = [2 * (im / 255.) - 1. for im in images]
        L = len(images)
        dflows, dflows_low = [], []
        flow_pre = None
        with torch.no_grad():
            for i in range(L - 1):
                dflow_low, dflow = self.ofe(images[i],
                                            images[i + 1],
                                            iters=iters,
                                            flow_init=flow_pre,
                                            test_mode=True)
                flow_pre = Softsplat('average')(dflow_low, dflow_low)
                dflows_low.append(dflow_low)
                dflows.append(dflow)

        # F01, F12, F23, ...
        assert len(dflows_low) == len(images) - 1

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            dfs = [self.fnet(x) for x in dflows_low]

        f = dfs[0]
        flows_all = []
        F0N = dflows_low[0]
        c = self.cnet(images[0])
        for i in range(1, L - 1):
            # F0N = F(0, i)
            F0N, _, emap = self.ofe(images[0],
                                    images[i + 1],
                                    iters=iters,
                                    flow_init=F0N,
                                    return_emap=True)
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                f_ofe = self.fnet(F0N)
                f = self.flowacc(f, dfs[i], c, f_ofe, emap)
                F0N = self.flow_head(f)
                m = self.mask_head(f)
            F0N_up = self.ofe.upsample_flow(F0N, m)

            flows_all.append(F0N_up)

        if test_mode:
            return [dflows[0]] + flows_all
        # F01, F02, F03, ...
        return flows_all
