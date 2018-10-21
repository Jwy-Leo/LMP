from __future__ import print_function
import math
import torch
import torch.nn as nn
import numpy 
import pdb

def mean_norm(x):

    batch_size = x.size(0)
    n = x.size(1) * x.size(2) * x.size(3)
    mean = torch.mean(x.view(batch_size, -1), 1)
    return \
        x.view(batch_size, -1).div(mean.expand(n, batch_size).transpose(0,1)).view(x.size())


class CBR(nn.Module):
    def __init__(self, inplanes, planes, kernel, stride=1, padding=0):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel, stride, padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DeCBR(nn.Module):
    def __init__(self, inplanes, planes, kernel, stride=1, padding=0):
        super(DeCBR, self).__init__()
        self.deconv = nn.ConvTranspose2d(inplanes, planes, kernel, stride,
                                         padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UmboUNet(nn.Module):
    def __init__(self, n_class=2, base_planes=32):
        super(UmboUNet, self).__init__()

        # the number of planes for encoder layers
        n_p = {'nP1': base_planes,
               'nP2': int(base_planes * math.pow(2, 1)),
               'nP4': int(base_planes * math.pow(2, 2)),
               'nP8': int(base_planes * math.pow(2, 3)),
               'nP16': int(base_planes * math.pow(2, 4))}

        # The number of planes for decoder layers
        n_p['nP12'] = n_p['nP4'] + n_p['nP8']
        n_p['nP6'] = n_p['nP2'] + n_p['nP4']
        n_p['nP5'] = n_p['nP1'] + n_p['nP4']

        self.seq5e = nn.Sequential(
            CBR(n_p['nP8'], n_p['nP16'], 4, 2, 1),
            CBR(n_p['nP16'], n_p['nP16'], 3, 1, 1),
            CBR(n_p['nP16'], n_p['nP16'], 3, 1, 1)
        )

        self.seq5d = nn.Sequential(
            CBR(n_p['nP16'], n_p['nP16'], 3, 1, 1),
            CBR(n_p['nP16'], n_p['nP16'], 3, 1, 1),
            DeCBR(n_p['nP16'], n_p['nP8'], 2, 2, 0)
        )

        self.seq4e = nn.Sequential(
            CBR(n_p['nP4'], n_p['nP8'], 4, 2, 1),
            CBR(n_p['nP8'], n_p['nP8'], 3, 1, 1),
            CBR(n_p['nP8'], n_p['nP8'], 3, 1, 1),
            CBR(n_p['nP8'], n_p['nP8'], 3, 1, 1)
        )

        self.seq4d = nn.Sequential(
            CBR(n_p['nP16'], n_p['nP16'], 1, 1, 0),
            CBR(n_p['nP16'], n_p['nP16'], 3, 1, 1),
            DeCBR(n_p['nP16'], n_p['nP8'], 2, 2, 0)
        )

        self.seq3e = nn.Sequential(
            CBR(n_p['nP2'], n_p['nP4'], 4, 2, 1),
            CBR(n_p['nP4'], n_p['nP4'], 3, 1, 1),
            CBR(n_p['nP4'], n_p['nP4'], 3, 1, 1),
            CBR(n_p['nP4'], n_p['nP4'], 3, 1, 1)
        )

        self.seq3d = nn.Sequential(
            CBR(n_p['nP12'], n_p['nP12'], 1, 1, 0),
            CBR(n_p['nP12'], n_p['nP12'], 3, 1, 1),
            DeCBR(n_p['nP12'], n_p['nP6'], 2, 2, 0)
        )

        self.seq2e = nn.Sequential(
            CBR(n_p['nP1'], n_p['nP2'], 4, 2, 1),
            CBR(n_p['nP2'], n_p['nP2'], 3, 1, 1),
            CBR(n_p['nP2'], n_p['nP2'], 3, 1, 1)
        )

        self.seq2d = nn.Sequential(
            CBR(n_p['nP8'], n_p['nP8'], 1, 1, 0),
            CBR(n_p['nP8'], n_p['nP8'], 3, 1, 1),
            DeCBR(n_p['nP8'], n_p['nP4'], 2, 2, 0)
        )

        self.seq1e = nn.Sequential(
            CBR(3, n_p['nP1'], 3, 1, 1),
            CBR(n_p['nP1'], n_p['nP1'], 3, 1, 1),
            CBR(n_p['nP1'], n_p['nP1'], 3, 1, 1)
        )

        self.seq1d = nn.Sequential(
            CBR(n_p['nP5'], n_p['nP5'], 1, 1, 0),
            CBR(n_p['nP5'], n_p['nP5'], 3, 1, 1),
            nn.Conv2d(n_p['nP5'], n_class, 1, 1, 0, bias=False)
        )

    def forward(self, x):

        # Encoding
        x1 = self.seq1e(x)
        x2 = self.seq2e(x1)
        x3 = self.seq3e(x2)
        x4 = self.seq4e(x3)
        x5 = self.seq5e(x4)

        # Decoding
        x6 = torch.cat((mean_norm(self.seq5d(x5)), mean_norm(x4)), 1)
        x7 = torch.cat((mean_norm(self.seq4d(x6)), mean_norm(x3)), 1)
        x8 = torch.cat((mean_norm(self.seq3d(x7)), mean_norm(x2)), 1)
        x9 = torch.cat((mean_norm(self.seq2d(x8)), mean_norm(x1)), 1)
        x10 = self.seq1d(x9)

        return x5,x10

