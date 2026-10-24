# CUHK-X Zenodo 19144427; extracted TNet and RadarPointNet, no training side effects.
# See third_party/NOTICE.md.
import torch
from torch import nn

class TNet(nn.Module):
    """Spatial Transformer Network"""
    def __init__(self, k=6):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc = nn.Linear(1024, k*k)
        self.k = k

    def forward(self, x):
        x = x.transpose(2, 1)
        bs = x.size(0)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max(self.conv3(x), 2, keepdim=True)[0]
        x = self.fc(x.view(bs, -1))
        iden = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(bs, 1)
        return x.view(bs, self.k, self.k) + iden.view(bs, self.k, self.k)

class RadarPointNet(nn.Module):
    """Enhanced PointNet (adapted for 4 input features)"""
    def __init__(self):
        super().__init__()
        self.input_tnet = TNet(k=4)  # Changed to 4-dimensional input
        self.mlp = nn.Sequential(
            nn.Conv1d(4, 64, 1),    # Input channels changed to 4
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, T, N, 4]
        B, T, N, _ = x.shape
        x = x.view(B*T, N, 4)  # Adjusted to 4 features

        # Apply spatial transformation (consistent with input dimension)
        trans = self.input_tnet(x)
        x = torch.bmm(x, trans)

        # Feature extraction (input dimension matched)
        x = x.transpose(2, 1)  # [B*T, 4, N]
        x = self.mlp(x)        # [B*T, 256, N]
        global_feat = torch.max(x, 2)[0]
        return global_feat.view(B, T, -1)
