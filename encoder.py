""" Implementation of PointNet encoder used in AtlasNet [1].

[1] Groueix Thibault et al. AtlasNet: A Papier-Mâché Approach to Learning 3D
    Surface Generation. CVPR 2018.

Authors:
(original) Thibault Groueix, thibault.groueix.2012@polytechnique.org
(adapted) Jan Bednarik, jan.bednarik@epfl.ch
"""

# 3rd party
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Project files.
from helpers import Device


#UTILITIES
class STN3d(nn.Module, Device):
    def __init__(self, gpu=True):
        nn.Module.__init__(self)
        Device.__init__(self, gpu=gpu)

        # self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self._iden = torch.from_numpy(np.eye(3, dtype=np.float32)).\
            reshape((1, 9)).to(self.device)

        self = self.to(self.device)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = self._iden.repeat(batchsize, 1)

        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module, Device):
    def __init__(self, global_feat=True, trans=False,
                 gpu=True):
        nn.Module.__init__(self)
        Device.__init__(self, gpu=gpu)

        super(PointNetfeat, self).__init__()
        self.stn = STN3d(gpu=gpu)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans

        # self.num_points = num_points
        self.global_feat = global_feat

        self = self.to(self.device)

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): Input pcloud, shape (B, N, 3).

        Returns:

        """

        # Adapt input shape (B, N, 3) to (B, 3, N) for nn.Conv1D to work.
        x = x.transpose(2, 1)

        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                raise NotImplementedError
        else:
            return x


class ANEncoderPN(nn.Module, Device):
    """ PointNet-based encoder used by AtlasNet.

    Args:
        N (int):
        code (int): Size of the CW.
        normalize_cw (bool): Whether to normalize the CW.
    """
    def __init__(self, code, normalize_cw=False, gpu=True):
        nn.Module.__init__(self)
        Device.__init__(self, gpu=gpu)

        self._normalize_cw = normalize_cw

        self.layers = nn.Sequential(
            PointNetfeat(global_feat=True, trans=False),
            nn.Linear(1024, code),
            nn.BatchNorm1d(code),
            nn.ReLU())

        self = self.to(self.device)

    def forward(self, pcloud):
        cws = self.layers(pcloud)

        if self._normalize_cw:
            cws = F.normalize(cws, dim=1)

        return cws