import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./models')
from pointnet_util import PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self,feature_size=128,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.projection_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, feature_size)
        )

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        h = l3_points.view(B, 1024)
        z = self.projection_head(h)

        return h, z

class ClassifierHead(nn.Module):
    def __init__(self, args, input_dim=1024, num_classes=40):
        nn.Module.__init__(self)

        self.fc = nn.Linear(input_dim, num_classes)
        self.fc = self.fc.cuda() if args.use_cuda else self.fc

    def forward(self, x):
        x = self.fc(x)
        return x